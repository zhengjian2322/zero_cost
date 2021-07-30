import pickle
import random
import ConfigSpace as CS
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter

darts_choices = [
    # 'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

benchmark201_choices = [
    'none',
    'skip_connect',
    'nor_conv_1x1',
    'nor_conv_3x3',
    'avg_pool_3x3'
]
ASR_main_edge_choices = [
    'linear',
    'conv5',
    'conv5d2',
    'conv7',
    'conv7d2',
    'zero'
]

ASR_skip_edge_choices = [
    'Zero',  # branch not present
    'Identity'  # branch present
]


def get_darts_config_space(blocks_in_cell=4):
    cs = ConfigurationSpace()
    # Normal Cell
    for i in range(blocks_in_cell * 2):
        normal_op_hp = CategoricalHyperparameter('normal_op%d' % i, darts_choices,
                                                 default_value=random.choice(darts_choices))
        cs.add_hyperparameter(normal_op_hp)
    # Reduction Cell
    for i in range(blocks_in_cell * 2):
        reduce_op_hp = CategoricalHyperparameter('reduce_op%d' % i, darts_choices,
                                                 default_value=random.choice(darts_choices))
        cs.add_hyperparameter(reduce_op_hp)

    for i in range(blocks_in_cell * 2):
        normal_connection_hp = CategoricalHyperparameter('normal_connection%d' % i, list(range(i // 2 + 2)),
                                                         default_value=random.choice(range((i // 2) + 2)))
        cs.add_hyperparameter(normal_connection_hp)
    for i in range(blocks_in_cell * 2):
        reduce_connection_hp = CategoricalHyperparameter('reduce_connection%d' % i, list(range(i // 2 + 2)),
                                                         default_value=random.choice(range((i // 2) + 2)))
        cs.add_hyperparameter(reduce_connection_hp)
    return cs


def check_benchmark201_config(archs, config):
    arch = '|%s~0|+|%s~0|%s~1|+|%s~0|%s~1|%s~2|' % (config['op_0'],
                                                    config['op_1'], config['op_2'],
                                                    config['op_3'], config['op_4'], config['op_5'])
    if arch in archs:
        return True
    return False


def get_benchmark201_config_space(operation=6):
    cs = ConfigurationSpace()
    for i in range(operation):
        cs.add_hyperparameter(
            CategoricalHyperparameter('op_%d' % i, choices=benchmark201_choices, default_value=benchmark201_choices[1]))

    return cs


def get_benchmarkASR_config_space(node_num=4):
    cs = ConfigurationSpace()

    # first node not have any edge
    for main_edge_id in range(1, node_num):
        cs.add_hyperparameter(
            CategoricalHyperparameter('main_edge_%d' % main_edge_id, choices=ASR_main_edge_choices,
                                      default_value=ASR_main_edge_choices[0]))
        for skip_edge_id in range(main_edge_id):
            cs.add_hyperparameter(CategoricalHyperparameter('main_edge_%d_%d' % (main_edge_id, skip_edge_id),
                                      choices=range(len(ASR_skip_edge_choices)),
                                      default_value=0))
    return cs
