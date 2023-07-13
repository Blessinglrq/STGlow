'''
Defined classes:
    class Normflow()
'''
import copy
from termcolor import colored
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from math import log, pi, exp
import nflows.utils.typechecks as check

logabs = lambda x: torch.log(torch.abs(x))


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                bias = torch.diag(torch.zeros_like(W[0])+1e-6)
                W_inverse = (W+bias).float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W


class GroupInstanceNorm(nn.Module):
    def __init__(self, features=None, group_len=5):
        """
        Args:
            group_len: the len of group
            tensor: shape of (B, L, K), K for the number of trajectory
        Returns: shape of (B, L, K)
        """
        if not check.is_positive_int(group_len):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()
        self.group_len = group_len
        self.features = features
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        # self.scale = nn.Parameter(torch.zeros(group_len))
        if features is not None:
            self.log_scale = nn.Parameter(torch.zeros(features, group_len))
            self.shift = nn.Parameter(torch.zeros(features, group_len))
        else:
            self.log_scale = nn.Parameter(torch.zeros(group_len))
            self.shift = nn.Parameter(torch.zeros(group_len))

    @property
    def scale(self):
        return torch.exp(torch.clamp(self.log_scale, max=1e10))

    def forward(self, inputs):
        if inputs.dim() != 3:
            raise ValueError("Expecting inputs to be a 3D tensor.")
        if inputs.dim() == 3:
            K_sample = inputs.shape[2]
            if K_sample % self.group_len != 0:
                raise ValueError("Expecting K_sample to be divisible by group_len.")
            B, C, K = inputs.shape
            if self.features is not None:
                inputs = inputs.unfold(2, self.group_len, self.group_len)
            else:
                inputs = inputs.reshape(-1, K)
                inputs = inputs.unfold(1, self.group_len, self.group_len)

        if self.training and not self.initialized:
            self._initialize(inputs)

        if self.features is not None:
            scale = self.scale.view(C, 1, -1).repeat(1, inputs.shape[2], 1)
            shift = self.shift.view(C, 1, -1).repeat(1, inputs.shape[2], 1)
            outputs = scale * inputs + shift
            outputs = outputs.reshape(B, C, -1)
        else:
            scale = self.scale.view(1, -1).repeat(inputs.shape[1], 1)
            shift = self.shift.view(1, -1).repeat(inputs.shape[1], 1)
            outputs = scale * inputs + shift
            outputs = outputs.reshape(-1, K).reshape(-1, C, K)

        logabsdet = torch.sum(self.log_scale)

        return outputs, logabsdet

    def inverse(self, inputs):
        if inputs.dim() != 3:
            raise ValueError("Expecting inputs to be a 3D tensor.")
        if inputs.dim() == 3:
            K_sample = inputs.shape[2]
            if K_sample % self.group_len != 0:
                raise ValueError("Expecting K_sample to be divisible by group_len.")
            B, C, K = inputs.shape
            if self.features is not None:
                inputs = inputs.unfold(2, self.group_len, self.group_len)
            else:
                inputs = inputs.reshape(-1, K)
                inputs = inputs.unfold(1, self.group_len, self.group_len)

        if self.features is not None:
            scale = self.scale.view(C, 1, -1).repeat(1, inputs.shape[2], 1)
            shift = self.shift.view(C, 1, -1).repeat(1, inputs.shape[2], 1)
            outputs = (inputs - shift) / (scale + 1e-6)
            outputs = outputs.reshape(B, C, -1)
        else:
            scale = self.scale.view(1, -1).repeat(inputs.shape[1], 1)
            shift = self.shift.view(1, -1).repeat(inputs.shape[1], 1)
            outputs = (inputs - shift) / (scale + 1e-6)
            outputs = outputs.reshape(-1, K).reshape(-1, C, K)

        return outputs

    def _initialize(self, inputs):
        """Data-dependent initialization"""
        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / (std + 1e-6)).mean(dim=0)
            std = std.mean(-2)
            mu = mu.mean(-2)
            self.log_scale.data = -torch.log(torch.clamp(std, 1e-10, 1e10))
            self.shift.data = -mu
            self.initialized.data = torch.tensor(True, dtype=torch.bool)


class TemporalGraphormer(torch.nn.Module):
    def __init__(self, input_len, t_head, t_layer, n_group, cfg, input_dim=None):
        super(TemporalGraphormer, self).__init__()
        self.t_head = t_head
        self.t_layer = t_layer
        self.n_group = n_group
        self.input_len = input_len
        self.input_dim = input_dim if input_dim is not None else 2
        self.cfg = cfg
        self.mlp = nn.Sequential(nn.Linear(
            self.input_dim, self.n_group),
            nn.GELU())
        self.position = nn.Parameter(torch.randn(self.input_len, self.n_group))
        self.encode_temporal_centrality = nn.Sequential(nn.Linear(
            1, self.n_group),
            nn.GELU())
        x_size = self.n_group
        self.temporal_graphformer = nn.ModuleList([EncoderLayer(x_size, ffn_size=self.n_group * 2,
                                                                dropout_rate=0.1, attention_dropout_rate=0.1,
                                                                num_heads=self.t_head, cfg=self.cfg) for _ in range(self.t_layer)])
    def forward(self, embedding):
        # 1. embedding
        ped_encoded = self.mlp(embedding) + self.position
        # 2. centrality encoded
        centrality = torch.arange(self.input_len - 1, -1, -1).float().to(ped_encoded.device)
        centrality_encoded = self.encode_temporal_centrality(centrality.reshape(-1, 1))
        # 3. mask multiheadattention
        ped_encoded = ped_encoded + centrality_encoded
        mask = torch.tril(torch.ones(self.input_len, self.input_len), diagonal=0).unsqueeze(0).repeat(
            self.t_head, 1, 1).to(ped_encoded.device)
        for module in self.temporal_graphformer:
            ped_encoded = module(ped_encoded, mask=mask)
        return ped_encoded[:, -1]


class GlowLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super(GlowLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output):
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]
        # FIXME way 1
        # loss = torch.sum(z*z)/(2*self.sigma*self.sigma) - log_s_total - log_det_W_total
        # FIXME way 2
        log_p = -0.5 * log(2*pi) - 0.5 * (z ** 2)
        log_p_sum = torch.sum(log_p)
        loss = - (log_p_sum + log_s_total + log_det_W_total)
        return loss/(z.size(0)*z.size(1)*z.size(2))


class TraFS(torch.nn.Module):
    def __init__(self, n_group, n_remaining_channels):
        super(TraFS, self).__init__()
        # encoder
        # self.z_embed = nn.Sequential(nn.Linear(n_group + n_remaining_channels // 2, 256),
        #                              # nn.ReLU()
        #                              nn.GELU(),
        #                              )
        self.z_embed = nn.Sequential(nn.Linear(n_group+n_remaining_channels//2, 2 * n_group),
                                       # nn.ReLU()
                                     # nn.BatchNorm1d(2 * n_group),
                                     nn.GELU(),
                                     nn.Linear(2 * n_group, 256),
                                     # nn.BatchNorm1d(256),
                                     nn.GELU(),
                                     )
        for m in self.z_embed.modules():
            if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.05)
                    m.bias.data.zero_()
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        # end = torch.nn.Conv1d(output_dim, input_dim*2//3, 1)
        # end = nn.Sequential(nn.Linear(output_dim, input_dim*2//3),
        #                                nn.ReLU())
        # end = nn.Linear(256, n_remaining_channels)
        # end.weight.data.zero_()
        # end.bias.data.zero_()
        # end = nn.Sequential(nn.Linear(256, n_remaining_channels),
        #                     nn.Sigmoid()
        #                     # nn.Tanh()
        #                     )
        end = nn.Linear(256, n_remaining_channels)
        # end = nn.Conv1d(256, n_remaining_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        # self.scale = nn.Parameter(torch.zeros(1, n_remaining_channels, 1))
        # self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # for m in end.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data.zero_()
        #         m.bias.data.zero_()
        self.end = end

    def forward(self, input_x):
        '''
        Params:
            input_x: (batch_size, segment_len, dim =2 or 4)
            target_y: (batch_size, pred_len, dim = 2 or 4)
        Returns:
            pred_traj: (batch_size, K, pred_len, 2 or 4)
        '''
        h_x = self.z_embed(input_x.permute(0, 2, 1))
        # B, C, L = input_x.shape  # 5.22
        # input_x = input_x.permute(0, 2, 1).reshape(-1, C)  # 5.22
        # h_x = self.z_embed(input_x).reshape(B, L, -1)  # 5.22
        h_x = self.end(h_x).permute(0, 2, 1)  # 5.22
        # h_x = self.z_embed(input_x.squeeze())  # 5.07
        # h_x = self.end(h_x).unsqueeze(2)  # 5.07
        # h_x = self.end(h_x.permute(0, 2, 1))
        # h_x = h_x * torch.exp(self.scale * 3)
        # h_x = torch.clamp(h_x, min=-10, max=10)
        # h_x = self.sigmoid(h_x)
        h_x = self.tanh(h_x)
        return h_x


class TraF(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TraF, self).__init__()
        # encoder
        self.z_embed = nn.Sequential(nn.Linear(input_dim, output_dim),
                                       nn.LeakyReLU())
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        # end = torch.nn.Conv1d(output_dim, input_dim*2//3, 1)
        # end = nn.Sequential(nn.Linear(output_dim, input_dim*2//3),
        #                                nn.ReLU())
        end = nn.Sequential(nn.Linear(output_dim, input_dim*2//3),
                      nn.Sigmoid())
        for m in end.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()
        # end = nn.Linear(output_dim, input_dim*2//3)
        # end.weight.data.zero_()
        # end.bias.data.zero_()
        self.end = end

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


class BiTrapSGlow(torch.nn.Module):
    def __init__(self, cfg, dataset_name=None, n_flows=16, n_group=256, sigma=1.0):
        super(BiTrapSGlow, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.K = self.cfg.K
        self.pred_len = self.cfg.PRED_LEN
        self.his_len = self.cfg.GLOBAL_INPUT_DIM
        self.dec_len = self.cfg.DEC_OUTPUT_DIM
        self.n_flow = n_flows
        self.n_group = n_group
        self.sigma = sigma

        self.n_early_every = 4
        self.n_early_size = self.n_group // self.n_early_every

        self.convinv = torch.nn.ModuleList()
        self.actnorm = torch.nn.ModuleList()
        self.instancenorm = torch.nn.ModuleList()

        self.trap_both = torch.nn.ModuleList()

        self.trap_his = TemporalGraphormer(input_len=self.cfg.INPUT_LEN, input_dim=self.his_len, t_head=8, t_layer=6, n_group=self.n_group, cfg=self.cfg)
        self.trap_fur = TemporalGraphormer(input_len=self.cfg.INPUT_LEN + self.pred_len, input_dim=self.dec_len, t_head=8, t_layer=6, n_group=self.n_group, cfg=self.cfg)

        n_half = int(n_group / 2)

        n_remaining_channels = n_group
        for i in range(n_flows):
            if i % self.n_early_every == 0 and i > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.instancenorm.append(GroupInstanceNorm(n_remaining_channels, group_len=20))  # Pattern Normalization
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            if self.cfg.GRAPHORMER_ENCODE:
                self.centrality = False
                self.spatial = True
                self.edge = True
                self.type_num = int(self.spatial + self.edge)
                self.trap_both.append(TraFS(self.n_group, n_remaining_channels))
            else:
                self.trap_both.append(TraFS(self.n_group, n_remaining_channels))

        self.n_remaining_channels = n_remaining_channels
        if self.cfg.PT_DIR:
            glow = torch.load(self.cfg.PT_DIR)
            self.load_state_dict(glow.state_dict())
            print(colored('Loaded checkpoint:{}'.format(cfg.FT_DIR), 'blue', 'on_green'))
        # goal predictor
        self.goal_decoder = nn.Sequential(nn.Linear(self.n_group, 128),
                                          nn.GELU(),
                                          nn.Linear(128, 64),
                                          nn.GELU(),
                                          nn.Linear(64, self.dec_len))
        self.enc_h_to_forward_h = nn.Sequential(nn.Linear(self.n_group, 256),
                                                nn.GELU(),
                                                )

        self.enc_h_to_backward_h = nn.Sequential(nn.Linear(self.n_group, 256),
                                                 nn.GELU(),
                                                )
        self.traj_dec_input_forward = nn.Sequential(nn.Linear(256, 256),
                                                    nn.GELU(),
                                                    )

        self.traj_dec_input_backward = nn.Sequential(nn.Linear(self.dec_len, 256),
                                                     nn.GELU(),
                                                     )
        self.traj_dec_forward = nn.GRUCell(input_size=256,
                                           hidden_size=256)

        self.traj_dec_backward = nn.GRUCell(input_size=256,
                                            hidden_size=256)
        self.traj_output = nn.Linear(256 * 2, self.dec_len)

        self.forward_traj_output = nn.Linear(512, self.dec_len)
        self.backward_traj_output = nn.Linear(512, self.dec_len)
        if self.cfg.GRAPHORMER_ENCODE:
            self.t_p = TemporalGraphormer(input_len=self.cfg.INPUT_LEN, input_dim=self.his_len, t_head=8, t_layer=6, n_group=self.n_group, cfg=self.cfg)
            if self.centrality:
                self.encode_node_centrality = nn.Sequential(nn.Linear(
                    1, self.n_group),
                    nn.GELU())
            if self.spatial:
                self.encode_node_spatial = nn.Sequential(nn.Linear(
                    2, self.n_group),
                    nn.GELU())
            if self.edge:
                self.encode_node_edge = nn.Sequential(nn.Linear(
                    1, self.n_group),
                    nn.GELU())
            x_size = self.n_group
            self.graphformer = nn.ModuleList([EncoderLayer(x_size, ffn_size=self.n_group*2,
                                     dropout_rate=0.1, attention_dropout_rate=0.1,
                                     num_heads=4) for _ in range(1)])
        self.criterion = GlowLoss(sigma)

    def forward(self, history, future, cur_pos=None, neighbors=None, neighbors_st=None, neigh_num=None, heading=None, first_history_indices=None, is_rel=False):
        cur_pos = history[:, -1, :] if cur_pos is None else cur_pos
        history_enc = self.trap_his(history)
        target_dir = history[:, -1] - history[:, -2]
        if neighbors_st is not None and self.cfg.GRAPHORMER_ENCODE:
            total_node_encoding = self.graphformer_encoder(neighbors, neighbors_st, cur_pos, neigh_num, heading, target_dir)
            history_enc = history_enc + total_node_encoding
        history_enc = history_enc.unsqueeze(2).repeat(1, 1, self.K)  # batch, 1, 2
        total_traj = torch.cat([history[:, :, :2], future], 1)
        future_enc = self.trap_fur(total_traj).unsqueeze(2).repeat(1, 1, self.K)  # batch, 1, 2
        future_enc, log_s_list, log_det_W_list = self.Glow(future_enc, history_enc)
        loss_flow = self.criterion([future_enc, log_s_list, log_det_W_list])
        # inverse
        _, loss_traj = self.infer(history, history_enc=history_enc, target=future, cur_pos=cur_pos, first_history_indices=first_history_indices, sigma=1.0, is_rel=is_rel)
        loss_dict = {'loss_flow': loss_flow, 'loss_traj': 0.5 * loss_traj['loss_traj'], 'loss_f_traj': 0.25 * loss_traj['loss_f_traj'],
                     'loss_b_traj': 0.25 * loss_traj['loss_b_traj'], 'loss_goal': loss_traj['loss_goal']}
        return loss_dict

    def infer(self, history, history_enc=None, target=None, z_sample_enc=None, cur_pos=None, neighbors=None, neighbors_st=None, neigh_num=None, heading=None, first_history_indices=None, sigma=1.0, is_rel=False, best_of_many=True, K_sample=None):  # add 5.12 sigma=0.1
        cur_pos = history[:, -1, :] if cur_pos is None else cur_pos
        K = self.K if K_sample is None else K_sample
        z_sample = torch.FloatTensor(history.size(0),
                                          self.n_remaining_channels,
                                          K).normal_().to(cur_pos.device)

        z_sample = torch.autograd.Variable(sigma*z_sample)
        if history_enc is None:
            history_enc = self.trap_his(history)  # batch, 1, 2
            target_dir = history[:, -1] - history[:, -2]
            if neighbors_st is not None and self.cfg.GRAPHORMER_ENCODE:
                total_node_encoding = self.graphformer_encoder(neighbors, neighbors_st, cur_pos, neigh_num, heading, target_dir)
                history_enc = history_enc + total_node_encoding
            history_enc = history_enc.unsqueeze(2).repeat(1, 1, K)  # batch, 1, 2
        # FIXME add interaction
        for k in reversed(range(self.n_flow)):
            n_half = int(z_sample.size(-2)/2)

            z_sample_0 = z_sample[:, :n_half, :]
            z_sample_1 = z_sample[:, n_half:, :]
            output = torch.cat([history_enc, z_sample_0], 1)
            output = self.trap_both[k](output)
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            z_sample_1 = (z_sample_1 - b)/torch.exp(s)
            z_sample = torch.cat([z_sample_0, z_sample_1], 1)
            z_sample = self.convinv[k](z_sample, reverse=True)
            z_sample = self.instancenorm[k].inverse(z_sample)
            if k % self.n_early_every == 0 and k > 0:
                z = torch.FloatTensor(history.size(0), self.n_early_size, K).normal_().to(history.device)
                z_sample = torch.cat((sigma*z, z_sample), 1)
        pred_goal = self.goal_decoder(z_sample.permute(0, 2, 1))
        future_rel, forward_traj, backward_traj = self.pred_future_traj_bi(z_sample.permute(0, 2, 1), pred_goal, K_sample=K)  # batch, frame, k, 2
        pred_goal = pred_goal + cur_pos.unsqueeze(1)
        cur_pos = history[:, None, -1, :] if cur_pos is None else cur_pos.unsqueeze(1)
        future = future_rel + cur_pos.unsqueeze(1)
        forward_future = forward_traj + cur_pos.unsqueeze(1)
        backward_future = backward_traj + cur_pos.unsqueeze(1)

        if target is not None:
            if is_rel:
                target = target + cur_pos
            target = target.unsqueeze(2).repeat(1, 1, K, 1)
            goal_rmse = torch.sqrt(torch.sum((pred_goal - target[:, -1, :, :]) ** 2, dim=-1) + 1e-8)
            traj_rmse = torch.sqrt(torch.sum((future - target) ** 2, dim=-1) + 1e-8).sum(dim=1)
            forward_traj_rmse = torch.sqrt(torch.sum((forward_future - target) ** 2, dim=-1) + 1e-8).sum(dim=1)
            backward_traj_rmse = torch.sqrt(torch.sum((backward_future - target) ** 2, dim=-1) + 1e-8).sum(dim=1)

            if best_of_many:
                best_idx = torch.argmin(traj_rmse, dim=1)
                loss_goal = goal_rmse[range(len(best_idx)), best_idx].mean()
                loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
                loss_f_traj = forward_traj_rmse[range(len(best_idx)), best_idx].mean()
                loss_b_traj = backward_traj_rmse[range(len(best_idx)), best_idx].mean()
            else:
                loss_goal = goal_rmse.mean()
                loss_traj = traj_rmse.mean()
                loss_f_traj = forward_traj_rmse.mean()
                loss_b_traj = backward_traj_rmse.mean()
            loss_dict = {'loss_traj': loss_traj, 'loss_f_traj': loss_f_traj, 'loss_b_traj': loss_b_traj,
                         'loss_goal': loss_goal}
            return future, loss_dict
        else:
            return future

    def pred_future_traj_bi(self, dec_h, G, K_sample=None):
        '''
        use a bidirectional GRU decoder to plan the path.
        Params:
            dec_h: (Batch, hidden_dim) if not using Z in decoding, otherwise (Batch, K, dim)
        Returns:
            backward_outputs: (Batch, T, K, pred_dim)
        '''
        forward_outputs = []
        forward_output = []
        backward_output = []
        K = self.K if K_sample is None else K_sample
        forward_h = self.enc_h_to_forward_h(dec_h)
        if len(forward_h.shape) == 2:
            forward_h = forward_h.unsqueeze(1).repeat(1, K, 1)
        forward_h = forward_h.view(-1, forward_h.shape[-1])
        forward_input = self.traj_dec_input_forward(forward_h)
        forward_input = forward_input.view(-1, forward_input.shape[-1])
        for t in range(self.pred_len):
            forward_h = self.traj_dec_forward(forward_input, forward_h)
            forward_input = self.traj_dec_input_forward(forward_h)
            forward_traj = self.forward_traj_output(torch.cat([forward_input, dec_h.reshape(-1, dec_h.shape[-1])], 1))
            forward_output.append(forward_traj.view(-1, K, forward_traj.shape[-1]))
            forward_outputs.append(forward_input)

        forward_outputs = torch.stack(forward_outputs, dim=1)
        forward_output = torch.stack(forward_output, dim=1)

        # 2. run backward on all samples
        backward_outputs = []

        backward_h = self.enc_h_to_backward_h(dec_h)
        if len(dec_h.shape) == 2:
            backward_h = backward_h.unsqueeze(1).repeat(1, K, 1)
        backward_h = backward_h.view(-1, backward_h.shape[-1])
        backward_input = self.traj_dec_input_backward(G)
        backward_input = backward_input.view(-1, backward_input.shape[-1])
        for t in range(self.pred_len - 1, -1, -1):
            backward_h = self.traj_dec_backward(backward_input, backward_h)
            backward_traj = self.backward_traj_output(torch.cat([backward_h, dec_h.reshape(-1, dec_h.shape[-1])], 1))
            output = self.traj_output(torch.cat([backward_h, forward_outputs[:, t]], dim=-1))
            backward_input = self.traj_dec_input_backward(output)
            backward_output.append(backward_traj.view(-1, K, backward_traj.shape[-1]))
            backward_outputs.append(output.view(-1, K, output.shape[-1]))

        # inverse because this is backward
        backward_outputs = backward_outputs[::-1]
        backward_output = backward_output[::-1]
        backward_outputs = torch.stack(backward_outputs, dim=1)
        backward_output = torch.stack(backward_output, dim=1)

        return backward_outputs, forward_output, backward_output

    def Glow(self, future_enc, history_enc):
        log_s_list = []
        log_det_W_list = []
        output_z = []
        for k in range(self.n_flow):
            if k % self.n_early_every == 0 and k > 0:
                output_z.append(future_enc[:, :self.n_early_size, :])
                future_enc = future_enc[:, self.n_early_size:, :]
            B, C, K = future_enc.shape
            future_enc, log_det_W_b = self.instancenorm[k](future_enc)
            log_det_W_b = log_det_W_b / K
            z, log_det_W_c = self.convinv[k](future_enc)
            log_det_W = log_det_W_c + log_det_W_b

            log_det_W_list.append(log_det_W)

            n_half = int(z.size(1) / 2)
            z_0 = z[:, :n_half, :]
            z_1 = z[:, n_half:, :]

            input_x = torch.cat([history_enc, z_0], 1)
            output = self.trap_both[k](input_x)
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            z_1 = torch.exp(log_s) * z_1 + b
            log_s_list.append(log_s)

            future_enc = torch.cat([z_0, z_1], 1)
        output_z.append(future_enc)
        future_enc = torch.cat(output_z, 1)
        return future_enc, log_s_list, log_det_W_list

    def graphformer_encoder(self, neighbors, neighbors_st, cur_pos, neigh_num, heading, target_dir):
        ##################################
        # 1. Encode History of neighbors #
        ##################################
        node_neighbors_encoded = self.t_p(neighbors_st)
        ##################################
        # 2. Encode centrality           #
        ##################################
        if self.centrality:
            node_centrality_encoded = self.encode_node_centrality(neigh_num.float().reshape(-1, 1))
        ##################################
        # 3. Spatial Encoding            #
        ##################################
        cur_pos_ = torch.repeat_interleave(cur_pos, neigh_num, dim=0)
        rel_pos = neighbors[:, -1, :2] - cur_pos_
        if self.spatial:
            node_spatial_encoded = self.encode_node_spatial(rel_pos)
        ##################################
        # 4. Edge Encoding               #
        ##################################
        if self.edge:
            node_edge_encoded = self.encode_node_edge(heading.reshape(-1, 1))
        ##########################
        # 5. Transformer Layers  #
        ##########################
        idx = 0
        interaction_encode = list()
        for scene in range(len(neigh_num)):
            scene_encode = node_neighbors_encoded[idx: idx+neigh_num[scene]]
            if self.centrality:
                scene_encode = scene_encode + node_centrality_encoded[0]
            if self.spatial and self.edge:
                scene_encode = scene_encode + node_spatial_encoded[idx: idx+neigh_num[scene]] + node_edge_encoded[
                                                                                            idx: idx+neigh_num[scene]]
            elif self.spatial:
                scene_encode = scene_encode + node_spatial_encoded[idx: idx+neigh_num[scene]]
            elif self.edge:
                scene_encode = scene_encode + node_edge_encoded[idx: idx+neigh_num[scene]]
            if neigh_num[scene] == 1:
                mask = None
            else:
                rel_scene = rel_pos[idx: idx+neigh_num[scene]]
                del_x = rel_scene[:, 0]
                del_y = rel_scene[:, 1]
                v_x = target_dir[scene, 0]
                v_y = target_dir[scene, 1]
                c = ((del_x * v_x >=0) * (del_y * v_y >=0))
                mask = torch.eye(neigh_num[scene]).to(rel_scene.device)
                mask[:, 0] = c
                mask[0] = torch.transpose(c.unsqueeze(1), 1, 0)
            idx += neigh_num[scene]
            scene_encode = scene_encode.unsqueeze(0)
            for module in self.graphformer:
                scene_encode = module(scene_encode, mask=mask)
            scene_encode = scene_encode.squeeze(0)
            interaction_encode.append(scene_encode[0])
        interaction_encode = torch.stack(interaction_encode, 0)
        return interaction_encode


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, cfg):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
        self.cfg = cfg

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, mask=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        seq_len = q.size(1)
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None and batch_size > 1:
            x[1:] = x[1:] + attn_bias.view(batch_size-1, -1, 1, 1)
        if mask is not None:
            x = x.mul(mask)
            # self.cfg.DEVICE = 'cuda:0'
            mask_ = ~(mask > 0.0) *(-1e9)
            # mask_ = torch.triu(torch.ones(mask.shape[1], mask.shape[1])*(-1e9), diagonal=1).to(self.cfg.DEVICE)
            x = x + mask_
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, seq_len, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, cfg=None):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, cfg)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, mask=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias, mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


if __name__ == '__main__':
    import argparse
    from bitrap.modeling import make_model
    from configs import cfg
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument(
        "--config_file",
        default="",
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
    model = make_model(cfg).to(cfg.DEVICE)
    # model.initialize()
    # history.shape: batch, his_frame, 6
    # future.shape: batch, fut_frame, 2
    history = torch.randn(128, 8, 6).to(cfg.DEVICE)
    cus = torch.randn(128, 2).to(cfg.DEVICE)
    future = torch.randn(128, 12, 2).to(cfg.DEVICE)
    a = model(history, future, cus)
    b = model.infer(history, target=future, cur_pos=cus)
    print(model)


