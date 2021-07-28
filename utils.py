from models import nasbench2
from models.darts.model import DartsNetworkCIFAR


def config_to_genotype(config, blocks_in_cell=4):
    from collections import namedtuple
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    # Normal Cell
    normal = list()
    for i in range(blocks_in_cell):
        normal.append((config['normal_op%d' % (2 * i)], config['normal_connection%d' % (2 * i)]))
        normal.append((config['normal_op%d' % (2 * i + 1)], config['normal_connection%d' % (2 * i + 1)]))
    normal_concat = [2, 3, 4, 5]
    # Reduction Cell
    reduce = list()
    for i in range(blocks_in_cell):
        reduce.append((config['reduce_op%d' % (2 * i)], config['reduce_connection%d' % (2 * i)]))
        reduce.append((config['reduce_op%d' % (2 * i + 1)], config['reduce_connection%d' % (2 * i + 1)]))
    reduce_concat = [2, 3, 4, 5]
    return Genotype(normal=normal, normal_concat=normal_concat, reduce=reduce, reduce_concat=reduce_concat)


def get_net_from_config(net_name, config, n_class, init_channels=16):
    if net_name == 'benchmark201':
        config = '|%s~0|+|%s~0|%s~1|+|%s~0|%s~1|%s~2|' % (config['op_0'],
                                                          config['op_1'], config['op_2'],
                                                          config['op_3'], config['op_4'], config['op_5'])
        net = nasbench2.get_model_from_arch_str(config, n_class, init_channels=init_channels)
    elif net_name == 'darts':
        genotype = config_to_genotype(config)
        net = DartsNetworkCIFAR(C=init_channels, num_classes=n_class,
                                layers=20, auxiliary=False, genotype=genotype)
        net.drop_path_prob = 0
    else:
        raise ValueError('%s is not supported' % net_name)
    return net
