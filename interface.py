import torch.cuda

from dataset.dataset import get_cifar_dataloaders
from pruners import predictive
from trian import load_arch_config_space
from utils import get_net_from_config


def get_nas_net_from_config(config, num_class, init_channels=16, arch_name='benchmark201'):
    net = get_net_from_config(arch_name, config, num_class, init_channels)

    return net


def get_zero_cost_metric(net, train_loader, num_class, measures, dataload='random', dataload_info=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    result = predictive.find_measures(net, train_loader,
                                      (dataload, dataload_info, num_class),
                                      device, measure_names=measures)
    return result


def example():
    train_loader, _ = get_cifar_dataloaders(train_batch_size=64, test_batch_size=64, dataset="cifar10",
                                            num_workers=0)
    archs_config_space = load_arch_config_space("benchmark201")
    archs_config = archs_config_space.sample_configuration(1)
    net = get_nas_net_from_config(archs_config, 10, 16, arch_name='benchmark201')
    measures = ['grad_norm', 'snip', 'grasp', 'fisher', 'jacob_cov', 'plain', 'synflow']

    print(get_zero_cost_metric(net, train_loader, 10, measures, dataload='random', dataload_info=1))


example()
