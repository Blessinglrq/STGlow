import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
import copy

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, n_of_group = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = n_of_group * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv1d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, n_of_group = input.shape

        out = F.conv1d(input, self.weight)
        logdet = (
            n_of_group * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv1d(
            output, self.weight.squeeze().inverse().unsqueeze(2)
        )


class InvConv1dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, n_of_group = input.shape

        weight = self.calc_weight()

        out = F.conv1d(input, weight)
        logdet = n_of_group * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv1d(output, weight.squeeze().inverse().unsqueeze(2))


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv1d(in_channel, out_channel, 1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1))

    def forward(self, input):
        # out = F.pad(input, [1, 1, 1], value=1)
        out = self.conv(input)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        # self.net = nn.Sequential(
        #     nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(filter_size, filter_size, 1),
        #     nn.ReLU(inplace=True),
        #     ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        # )
        self.trap_both = TraFS(in_channel)

        # self.net[0].weight.data.normal_(0, 0.05)
        # self.net[0].bias.data.zero_()
        #
        # self.net[2].weight.data.normal_(0, 0.05)
        # self.net[2].bias.data.zero_()

    def forward(self, input, his_enc):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            input_x = torch.cat([his_enc, in_b], 1)
            log_s, t = self.trap_both(input_x).chunk(2, 1)
            out_a = torch.exp(log_s) * in_a + t

            logdet = torch.sum(log_s.view(input.shape[0], -1), 1)

        else:
            input_x = torch.cat([his_enc, in_b], 1)
            net_out = self.trap_both(input_x)
            out_a = in_a + net_out
            logdet = None

        return torch.cat([out_a, in_b], 1), logdet

    def reverse(self, output, his_enc):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            input_x = torch.cat([his_enc, out_b], 1)
            log_s, t = self.trap_both(input_x).chunk(2, 1)
            # s = torch.exp(log_s)
            in_a = (out_a - t) / torch.exp(log_s)
            # s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            # in_b = out_b / s - t
        else:
            input_x = torch.cat([his_enc, out_b], 1)
            net_out = self.trap_both(input_x)
            in_a = out_b - net_out

        return torch.cat([in_a, out_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv1dLU(in_channel)

        else:
            self.invconv = InvConv1d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input, his_enc):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out, his_enc)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output, his_enc):
        input = self.coupling.reverse(output, his_enc)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, n_group, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = n_group

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv1d(n_group//2, n_group)

        else:
            self.prior = ZeroConv1d(n_group, n_group*2)

    def forward(self, input, history):
        b_size, n_channel, n_group = input.shape
        # squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        # squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        # out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
        out = input

        logdet = 0

        for flow in self.flows:
            out, det = flow(out, history)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, history_enc, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input, history_enc)

        b_size, n_channel, n_group = input.shape

        unsqueezed = input.view(b_size, n_channel, n_group)

        return unsqueezed


class Glow(nn.Module):
    def __init__(
        self, cfg, dataset_name=None, n_flow=32, n_group=256, n_block=4, affine=True, conv_lu=True
    ):
        super(Glow, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.K = self.cfg.K
        self.pred_len = self.cfg.PRED_LEN
        self.blocks = nn.ModuleList()
        self.n_flow = n_flow
        self.n_group = n_group
        self.n_block = n_block
        for i in range(self.n_block - 1):
            self.blocks.append(Block(self.n_group, self.n_flow, affine=affine, conv_lu=conv_lu))
            # self.n_group *= 2
        self.blocks.append(Block(self.n_group, self.n_flow, split=False, affine=affine))
        self.trap_his = TraP(input_dim=6, output_dim=self.n_group, cfg=self.cfg)
        self.trap_fur = TraP(input_dim=2, output_dim=self.n_group, cfg=self.cfg)

        self.goal_decoder = nn.Sequential(nn.Linear(self.n_group, 128),
                                          # nn.LeakyReLU(),
                                          nn.GELU(),
                                          nn.Linear(128, 64),
                                          # nn.LeakyReLU(),
                                          nn.GELU(),
                                          nn.Linear(64, self.cfg.DEC_OUTPUT_DIM))
        self.enc_h_to_forward_h = nn.Sequential(nn.Linear(self.n_group, 256),
                                                # nn.LeakyReLU(),
                                                nn.GELU(),
                                                )

        self.enc_h_to_backward_h = nn.Sequential(nn.Linear(self.n_group, 256),
                                                 # nn.LeakyReLU(),
                                                 nn.GELU(),
                                                 )
        self.traj_dec_input_forward = nn.Sequential(nn.Linear(256, 256),
                                                    # nn.LeakyReLU(),
                                                    nn.GELU(),
                                                    )

        self.traj_dec_input_backward = nn.Sequential(nn.Linear(2, 256),
                                                     # nn.LeakyReLU(),
                                                     nn.GELU(),
                                                     )
        self.traj_dec_forward = nn.GRUCell(input_size=256,
                                           hidden_size=256)

        self.traj_dec_backward = nn.GRUCell(input_size=256,
                                            hidden_size=256)
        self.traj_output = nn.Linear(256 * 2, 2)

    def forward(self, history, future, cur_pos=None, first_history_indices=None):
        cur_pos = history[:, -1, :] if cur_pos is None else cur_pos
        log_p_sum = 0
        logdet = 0
        z_outs = []

        history_enc = self.trap_his(history, first_history_indices)  # batch, 1, 2
        total_traj = torch.cat([history[:, :, :2], future], 1)
        future_enc = self.trap_fur(total_traj, first_history_indices)
        future_enc = future_enc.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        history_enc = history_enc.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        for block in self.blocks:
            out, det, log_p, z_new = block(future_enc, history_enc)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p
        logdet = logdet.mean()
        loss_flow, log_p, log_det = calc_loss(log_p_sum, logdet)
        _, loss_traj = self.infer(history, target=future, cur_pos=cur_pos, first_history_indices=first_history_indices)
        loss_dict = {'loss_flow': loss_flow, 'loss_traj': loss_traj['loss_traj'], 'loss_goal': loss_traj['loss_goal']}
        return loss_dict

    def infer(self, history, target=None, z_sample_enc=None, cur_pos=None, first_history_indices=None, sigma=1.0, best_of_many=True):
        cur_pos = history[:, -1, :] if cur_pos is None else cur_pos
        if z_sample_enc is None:
            z_sample = []
            z_shapes = calc_z_shapes(self.n_group, self.K, self.n_flow, self.n_block)
            for z in z_shapes:
                z_new = torch.randn(history.size(0), *z)
                z_new = torch.autograd.Variable(sigma * z_new)
                z_sample.append(z_new.cuda())
            # z_sample = torch.cuda.FloatTensor(history.size(0),
            #                                   self.n_group,
            #                                   self.K).normal_()
        else:
            temp = self.n_group - self.n_remaining_channels
            self.K = z_sample_enc.shape[2]
            z_sample = z_sample_enc[:, self.n_group - self.n_remaining_channels:, :]

        # z_sample = torch.autograd.Variable(sigma * z_sample)
        history_enc = self.trap_his(history, first_history_indices).unsqueeze(2).repeat(1, 1, self.K)  # batch, 1, 2
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_sample[-1], history_enc, z_sample[-1], reconstruct=False)

            else:
                input = block.reverse(input, history_enc, z_sample[-(i + 1)], reconstruct=False)

        pred_goal = self.goal_decoder(input.permute(0, 2, 1))
        future_rel = self.pred_future_traj_bi(input.permute(0, 2, 1), pred_goal)  # batch, frame, k, 2
        pred_goal = pred_goal + cur_pos.unsqueeze(1)
        cur_pos = history[:, None, -1, :] if cur_pos is None else cur_pos.unsqueeze(1)
        future = future_rel + cur_pos.unsqueeze(1)
        if target is not None:
            target = target.unsqueeze(2).repeat(1, 1, self.K, 1)
            # FIXME loss 1
            goal_rmse = torch.sqrt(torch.sum((pred_goal - target[:, -1, :, :]) ** 2, dim=-1))
            traj_rmse = torch.sqrt(torch.sum((future - target) ** 2, dim=-1)).sum(dim=1)
            # FIXME loss 2
            # goal_rmse = self.smoothl1loss(pred_goal, target[:, -1, :, :]).sum(-1)
            # traj_rmse = self.smoothl1loss(future, target).sum(-1).sum(1)
            # FIXME add refine
            # refine_traj_rmse = torch.sqrt(torch.sum((re_future - target) ** 2, dim=-1)).sum(dim=1)

            if best_of_many:
                best_idx = torch.argmin(traj_rmse, dim=1)
                # best_idx_rf = torch.argmin(refine_traj_rmse, dim=1)
                loss_goal = goal_rmse[range(len(best_idx)), best_idx].mean()
                loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
                # loss_rf_traj = refine_traj_rmse[range(len(best_idx_rf)), best_idx_rf].mean()
            else:
                loss_goal = goal_rmse.mean()
                loss_traj = traj_rmse.mean()
                # loss_rf_traj = refine_traj_rmse.mean()
            # loss_dict = {'loss_traj': loss_traj, 'loss_goal': loss_goal, 'loss_rf_traj': loss_rf_traj}
            loss_dict = {'loss_traj': loss_traj, 'loss_goal': loss_goal}
            return future, loss_dict
        else:
            return future


class TraP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, cfg):
        super(TraP, self).__init__()
        # encoder
        self.cfg = cfg
        self.box_embed = nn.Sequential(nn.Linear(input_dim, output_dim),
                                       # nn.ReLU()
                                       nn.GELU(),
                                       )
        self.box_encoder = nn.GRU(input_size=output_dim,
                                  hidden_size=output_dim,
                                  batch_first=True)

    def encode_variable_length_seqs(self, original_seqs, lower_indices=None, upper_indices=None, total_length=None):
        '''
        take the input_x, pack it to remove NaN, embed, and run GRU
        '''
        bs, tf = original_seqs.shape[:2]
        if lower_indices is None:
            lower_indices = torch.zeros(bs, dtype=torch.int)
        if upper_indices is None:
            upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
        if total_length is None:
            total_length = max(upper_indices) + 1
        # This is done so that we can just pass in self.prediction_timesteps
        # (which we want to INCLUDE, so this will exclude the next timestep).
        inclusive_break_indices = upper_indices + 1
        pad_list = []
        length_per_batch = []
        for i, seq_len in enumerate(inclusive_break_indices):
            pad_list.append(original_seqs[i, lower_indices[i]:seq_len])
            length_per_batch.append(seq_len - lower_indices[i])

        # 1. embed and convert back to pad_list
        x = self.box_embed(torch.cat(pad_list, dim=0))
        pad_list = torch.split(x, length_per_batch)

        # 2. run temporal
        packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False)
        packed_output, h_x = self.box_encoder(packed_seqs)
        # pad zeros to the end so that the last non zero value
        output, _ = rnn.pad_packed_sequence(packed_output,
                                            batch_first=True,
                                            total_length=total_length)
        return output, h_x

    def encoder(self, x, first_history_indices=None):
        '''
        x: encoder inputs
        '''
        outputs, _ = self.encode_variable_length_seqs(x,
                                                      lower_indices=first_history_indices)
        outputs = F.dropout(outputs,
                            p=self.cfg.DROPOUT,
                            training=self.training)
        if first_history_indices is not None:
            last_index_per_sequence = -(first_history_indices + 1)
            return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]
        else:
            # if no first_history_indices, all sequences are full length
            return outputs[:, -1, :]

    def forward(self, input_x, first_history_indices=None):
        '''
        Params:
            input_x: (batch_size, segment_len, dim =2 or 4)
            target_y: (batch_size, pred_len, dim = 2 or 4)
        Returns:
            pred_traj: (batch_size, K, pred_len, 2 or 4)
        '''
        h_x = self.encoder(input_x, first_history_indices)

        return h_x


class TraFS(torch.nn.Module):
    def __init__(self, n_group):
        super(TraFS, self).__init__()
        # encoder
        self.z_embed = nn.Sequential(nn.Linear(n_group+n_group//2, 256),
                                       # nn.ReLU()
                                     nn.GELU(),
                                     )
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        # end = torch.nn.Conv1d(output_dim, input_dim*2//3, 1)
        # end = nn.Sequential(nn.Linear(output_dim, input_dim*2//3),
        #                                nn.ReLU())
        # end = nn.Linear(256, n_remaining_channels)
        # end.weight.data.zero_()
        # end.bias.data.zero_()
        self.end = nn.Sequential(nn.Linear(256, n_group),
                            nn.Sigmoid()
                            # nn.Tanh()
                            )
        self.z_embed[0].weight.data.normal_(0, 0.05)
        self.z_embed[0].bias.data.zero_()

        self.end[0].weight.data.normal_(0, 0.05)
        self.end[0].bias.data.zero_()

    def forward(self, input_x):
        '''
        Params:
            input_x: (batch_size, segment_len, dim =2 or 4)
            target_y: (batch_size, pred_len, dim = 2 or 4)
        Returns:
            pred_traj: (batch_size, K, pred_len, 2 or 4)
        '''
        h_x = self.z_embed(input_x.permute(0, 2, 1))
        h_x = self.end(h_x).permute(0, 2, 1)
        return h_x


def calc_loss(log_p, logdet):
    # log_p = calc_log_p([z_list])
    loss = logdet + log_p

    return (
        (-loss).mean(),
        (log_p).mean(),
        (logdet).mean(),
    )

def calc_z_shapes(n_channel, K, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        z_shapes.append((n_channel//2, K))

    z_shapes.append((n_channel, K))

    return z_shapes

if __name__ == '__main__':
    import argparse
    from bitrap.modeling import make_model
    from configs import cfg
    parser = argparse.ArgumentParser(description="Glow trainer")
    parser.add_argument(
        "--n_flow", default=32, type=int, help="number of flows in each block"
    )
    parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
    parser.add_argument(
        "--no_lu",
        default=True,
        type=bool,
        help="use plain convolution instead of LU decomposed version",
    )
    parser.add_argument(
        "--affine", default=True, type=bool, help="use affine coupling instead of additive"
    )
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument(
        "--config_file",
        default="E:/paper/轨迹预测/bidireaction-trajectory-prediction/configs/stglow_ETH.yml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    history = torch.randn(128, 8, 6).cuda()
    cus = torch.randn(128, 2).cuda()
    future = torch.randn(128, 12, 2).cuda()
    # future = torch.randn(128, 20, 2).cuda()
    model = make_model(cfg).cuda()
    a = model(history, future, cus)
    b = model.infer(history, future, cur_pos=cus)
    print(model)