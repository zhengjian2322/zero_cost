import torch.cuda
import torch.nn.functional as F

from dataset.dataset import get_dataloaders
from pruners import predictive
from trian import load_arch_config_space
from utils import get_net_from_config


def get_nas_net_from_config(config, num_class, init_channels=16, arch_name='ASR'):
    net = get_net_from_config(arch_name, config, num_class, init_channels)

    return net


def loss(output, output_len, targets, targets_len):
    output_trans = output.permute(1, 0, 2)  # needed by the CTCLoss
    loss = F.ctc_loss(output_trans, targets, output_len, targets_len, reduction='none', zero_infinity=True)
    loss /= output_len
    loss = loss.mean()
    return loss


def get_zero_cost_metric(net, train_loader, num_class, measures, dataload='random', dataload_info=1, dataset='cifar10'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    if dataset == 'timit':

        result = predictive.find_measures(net, train_loader,
                                          (dataload, dataload_info, num_class),
                                          device, measure_names=measures, dataset=dataset,loss_fn=loss)
    else:
        result = predictive.find_measures(net, train_loader,
                                          (dataload, dataload_info, num_class),
                                          device, measure_names=measures, dataset=dataset)
    return result


def example(arch_name):
    train_loader, _ = get_dataloaders(train_batch_size=64, test_batch_size=64, dataset="TIMIT",
                                      num_workers=0)
    # traindata = []
    # dataloader_iter = iter(train_loader)
    #
    # traindata.append(next(dataloader_iter))
    # f = [a[0][0] for a in traindata]
    # inputs = torch.cat(f)
    # inputs_len=torch.cat([ a[1] for a in traindata[0]])

    archs_config_space = load_arch_config_space(arch_name)
    archs_config = archs_config_space.sample_configuration(1)
    net = get_nas_net_from_config(archs_config, 10, 16, arch_name=arch_name)
    # measures = ['grad_norm', 'snip', 'grasp', 'fisher', 'jacob_cov', 'plain', 'synflow']
    measures = ['grad_norm']

    print(get_zero_cost_metric(net, train_loader, 10, measures, dataload='random', dataload_info=1,dataset='timit'))


example('ASR')
