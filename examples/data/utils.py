import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def graph_matching_fusion(networks: list):
    n1 = total_node_num(network=networks[0])
    n2 = total_node_num(network=networks[1])
    assert (n1 == n2)
    affinity = torch.zeros([n1 * n2, n1 * n2], device=device)
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    num_nodes_before = 0
    num_nodes_incremental = []
    num_nodes_layers = []
    pre_conv_list = []
    cur_conv_list = []
    conv_kernel_size_list = []
    num_nodes_pre = 0
    is_conv = False
    pre_conv = False
    pre_conv_out_channel = 1
    is_final_bias = False
    perm_is_complete = True
    named_weight_list_0 = [named_parameter for named_parameter in networks[0].named_parameters()]
    for idx, ((_, fc_layer0_weight), (_, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        layer_shape = fc_layer0_weight.shape
        num_nodes_cur = fc_layer0_weight.shape[0]
        if len(layer_shape) > 1:
            if is_conv is True and len(layer_shape) == 2:
                num_nodes_pre = pre_conv_out_channel
            else:
                num_nodes_pre = fc_layer0_weight.shape[1]
        if idx >= 1 and len(named_weight_list_0[idx - 1][1].shape) == 1:
            pre_bias = True
        else:
            pre_bias = False
        if len(layer_shape) > 2:
            is_bias = False
            if not pre_bias:
                pre_conv = is_conv
                pre_conv_list.append(pre_conv)
            is_conv = True
            cur_conv_list.append(is_conv)
            fc_layer0_weight_data = fc_layer0_weight.data.view(
                fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(
                fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        elif len(layer_shape) == 2:
            is_bias = False
            if not pre_bias:
                pre_conv = is_conv
                pre_conv_list.append(pre_conv)
            is_conv = False
            cur_conv_list.append(is_conv)
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data
        else:
            is_bias = True
            if not pre_bias:
                pre_conv = is_conv
                pre_conv_list.append(pre_conv)
            is_conv = False
            cur_conv_list.append(is_conv)
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data
        if is_conv:
            pre_conv_out_channel = num_nodes_cur
        if is_bias is True and idx == num_layers - 1:
            is_final_bias = True
        if idx == 0:
            for a in range(num_nodes_pre):
                affinity[(num_nodes_before + a) * n2 + num_nodes_before + a] \
                    [(num_nodes_before + a) * n2 + num_nodes_before + a] \
                    = 1
        if idx == num_layers - 2 and 'bias' in named_weight_list_0[idx + 1][0] or \
                idx == num_layers - 1 and 'bias' not in named_weight_list_0[idx][0]:
            for a in range(num_nodes_cur):
                affinity[(num_nodes_before + num_nodes_pre + a) * n2 + num_nodes_before + num_nodes_pre + a] \
                    [(num_nodes_before + num_nodes_pre + a) * n2 + num_nodes_before + num_nodes_pre + a] \
                    = 1
        if is_bias is False:
            ground_metric = Ground_Metric_GM(
                fc_layer0_weight_data, fc_layer1_weight_data, is_conv, is_bias,
                pre_conv, int(fc_layer0_weight_data.shape[1] / pre_conv_out_channel))
        else:
            ground_metric = Ground_Metric_GM(
                fc_layer0_weight_data, fc_layer1_weight_data, is_conv, is_bias,
                pre_conv, 1)

        layer_affinity = ground_metric.process_soft_affinity(p=2)

        if is_bias is False:
            pre_conv_kernel_size = fc_layer0_weight.shape[3] if is_conv else None
            conv_kernel_size_list.append(pre_conv_kernel_size)
        if is_bias is True and is_final_bias is False:
            for a in range(num_nodes_cur):
                for c in range(num_nodes_cur):
                    affinity[(num_nodes_before + a) * n2 + num_nodes_before + c] \
                        [(num_nodes_before + a) * n2 + num_nodes_before + c] \
                        = layer_affinity[a][c]
        elif is_final_bias is False:
            for a in range(num_nodes_pre):
                for b in range(num_nodes_cur):
                    affinity[
                    (num_nodes_before + a) * n2 + num_nodes_before:
                    (num_nodes_before + a) * n2 + num_nodes_before + num_nodes_pre,
                    (num_nodes_before + num_nodes_pre + b) * n2 + num_nodes_before + num_nodes_pre:
                    (num_nodes_before + num_nodes_pre + b) * n2 + num_nodes_before + num_nodes_pre + num_nodes_cur] \
                        = layer_affinity[a + b * num_nodes_pre].view(num_nodes_cur, num_nodes_pre).transpose(0, 1)
        if is_bias is False:
            num_nodes_before += num_nodes_pre
            num_nodes_incremental.append(num_nodes_before)
            num_nodes_layers.append(num_nodes_cur)
    return affinity, [n1, n2, num_nodes_incremental, num_nodes_layers, cur_conv_list, conv_kernel_size_list]


def align(solution, ensemble_step, networks: list, params: list):
    [_, _, num_nodes_incremental, num_nodes_layers, cur_conv_list, conv_kernel_size_list] = params
    named_weight_list_0 = [named_parameter for named_parameter in networks[0].named_parameters()]
    aligned_wt_0 = [parameter.data for name, parameter in named_weight_list_0]
    idx = 0
    num_layers = len(aligned_wt_0)
    for num_before, num_cur, cur_conv, cur_kernel_size in \
            zip(num_nodes_incremental, num_nodes_layers, cur_conv_list, conv_kernel_size_list):
        perm = solution[num_before:num_before + num_cur, num_before:num_before + num_cur]
        assert 'bias' not in named_weight_list_0[idx][0]
        if len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = (perm.transpose(0, 1).to(torch.float64) @ \
                                 aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0, 1)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64)
        idx += 1
        if idx >= num_layers:
            continue
        if 'bias' in named_weight_list_0[idx][0]:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
            idx += 1
        if idx >= num_layers:
            continue
        if cur_conv and len(named_weight_list_0[idx][1].shape) == 2:
            aligned_wt_0[idx] = (aligned_wt_0[idx].to(torch.float64) \
                                 .reshape(aligned_wt_0[idx].shape[0], 64, -1) \
                                 .permute(0, 2, 1) \
                                 @ perm.to(torch.float64)) \
                .permute(0, 2, 1) \
                .reshape(aligned_wt_0[idx].shape[0], -1)
        elif len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = (aligned_wt_0[idx].to(torch.float64) \
                                 .permute(2, 3, 0, 1) \
                                 @ perm.to(torch.float64)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
    assert idx == num_layers

    averaged_weights = []
    for idx, parameter in enumerate(networks[1].parameters()):
        averaged_weights.append((1 - ensemble_step) * aligned_wt_0[idx] + ensemble_step * parameter)
    return averaged_weights


def total_node_num(network: torch.nn.Module):
    # count the total number of nodes in the network [network]
    num_nodes = 0
    for idx, (name, parameters) in enumerate(network.named_parameters()):
        if 'bias' in name:
            continue
        if idx == 0:
            num_nodes += parameters.shape[1]
        num_nodes += parameters.shape[0]
    return num_nodes


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=1, padding_mode='replicate', bias=False)
        self.max_pool = nn.MaxPool2d(2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1, padding_mode='replicate', bias=False)
        self.fc1 = nn.Linear(3136, 32, bias=False)
        self.fc2 = nn.Linear(32, 10, bias=False)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = self.max_pool(output)
        output = F.relu(self.conv2(output))
        output = self.max_pool(output)
        output = output.view(output.shape[0], -1)
        output = self.fc1(output)
        output = self.fc2(output)
        return output


class Ground_Metric_GM:
    def __init__(self,
                 model_1_param: torch.tensor = None,
                 model_2_param: torch.tensor = None,
                 conv_param: bool = False,
                 bias_param: bool = False,
                 pre_conv_param: bool = False,
                 pre_conv_image_size_squared: int = None):
        self.model_1_param = model_1_param
        self.model_2_param = model_2_param
        self.conv_param = conv_param
        self.bias_param = bias_param
        # bias, or fully-connected from linear
        if bias_param is True or (conv_param is False and pre_conv_param is False):
            self.model_1_param = self.model_1_param.reshape(1, -1, 1)
            self.model_2_param = self.model_2_param.reshape(1, -1, 1)
        # fully-connected from conv
        elif conv_param is False and pre_conv_param is True:
            self.model_1_param = self.model_1_param.reshape(1, -1, pre_conv_image_size_squared)
            self.model_2_param = self.model_2_param.reshape(1, -1, pre_conv_image_size_squared)
        # conv
        else:
            self.model_1_param = self.model_1_param.reshape(1, -1, model_1_param.shape[-1])
            self.model_2_param = self.model_2_param.reshape(1, -1, model_2_param.shape[-1])

    def process_distance(self, p: int = 2):
        return torch.cdist(
            self.model_1_param.to(torch.float),
            self.model_2_param.to(torch.float),
            p=p)[0]

    def process_soft_affinity(self, p: int = 2):
        return torch.exp(0 - self.process_distance(p=p))
