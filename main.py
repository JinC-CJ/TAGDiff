import argparse

from runner import Runner
from utils import parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='PreferDiff', help='Model name')
        
    # 新增 single_domain 参数
    parser.add_argument('-s', '--single_domain', type=str, default=None,
                        help='Single domain for both source and target (overrides --sd and --td)')
    
    # 保留原有参数以保持向后兼容
    parser.add_argument('--sd', type=str, default='C')
    parser.add_argument('--td', type=str, default='O')
    parser.add_argument('--exp_type', type=str, default='srec')
    parser.add_argument('--encoding_type', type=str, default='RFF')
    parser.add_argument('--sigma', type=str)
    parser.add_argument('--predict', type=str, default='N')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    args_dict = vars(args)
        
    # 如果指定了 single_domain，则覆盖 sd 和 td
    if args.single_domain:
        args_dict['sd'] = args.single_domain
        args_dict['td'] = args.single_domain
        args_dict['single_domain_mode'] = True  # 添加标记，表示使用单一数据集模式
    else:
        args_dict['single_domain_mode'] = False

    merged_dict = {**args_dict, **command_line_configs}

    runner = Runner(
        model_name=args.model,
        config_dict=merged_dict
    )
    runner.run()