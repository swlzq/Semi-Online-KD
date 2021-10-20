import argparse
import yaml
import os
import torch

from trainer import build_trainer
from utils.utils import save_code, save_opts


def main():
    parser = argparse.ArgumentParser(description='KnowledgeDistillation')
    parser.add_argument('--configs', '-c', dest='params', default='./configs/sokd.yaml')
    parser.add_argument('--name', '-n', dest='name', default='debug')
    parser.add_argument('--seed', '-s', type=int, default=8888)
    parser.add_argument('--gpus', '-g', type=str, default='0')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    params['name'] = args.name
    params['seed'] = args.seed
    params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = build_trainer(**params)
    save_opts(params, trainer.save_folder)
    save_code(trainer.repo_path, f"{trainer.save_folder}/code", ['results', 'datasets'])
    trainer.run()
    trainer.logger.info(f"{trainer.experimental_name} done!")
    

if __name__ == '__main__':
    main()
