from trainer.trainer_rdkit_descriptors import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import argparse
from torchsummary import summary
from tqdm.notebook import tqdm
import pandas as pd
import os
import subprocess
from distutils.util import strtobool
import shutil
from dotenv import load_dotenv
load_dotenv()

def main(opt):
    if opt.delete_logs and os.path.exists(opt.log_path):
        shutil.rmtree(opt.log_path)
    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)
    writer = SummaryWriter(log_dir=opt.log_path)
    subprocess.Popen('tensorboard --logdir {}'.format(opt.log_path), shell=True)

    trainer = Trainer(opt)
    for i in tqdm(range(opt.epoch)):
        loss, r2_train, r2_test = trainer.train()
        writer.add_scalar('loss/', loss, i)
        writer.add_scalar('r2_score_train/', r2_train, i)
        writer.add_scalar('r2_score_test/', r2_test, i)

    df_test = pd.read_csv('./neural_network/cache/test_{}.csv'.format(opt.y_column), index_col=0)
    x_test = df_test.iloc[:, 1:-1].values
    x_test = torch.FloatTensor(x_test)
    y_pred = trainer.network(x_test)
    y_pred = np.squeeze(y_pred.to('cpu').detach().numpy().copy())
    df_test['pred_{}'.format(opt.y_column)] = y_pred
    df_test.to_csv('./neural_network/cache/test_{}.csv'.format(opt.y_column))

    ### save model ###
    if not os.path.exists('./neural_network/model'):
        os.makedirs('./neural_network/model', exist_ok=True)
    model_path = './neural_network/model/model_{}.pth'.format(opt.y_column)
    torch.save(trainer.network.state_dict(), model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=10000, type=int, help="set your dataframe's path")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--log_path", default="./log/", type=str, help="log file path")
    parser.add_argument("--df_path", type=str, help="log file path")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="log file path")
    parser.add_argument("--y_column", type=str, help="log file path")
    parser.add_argument("--normalize", type=strtobool, default=False, help="正規化を行うか行わないか")
    parser.add_argument("--delete_logs", type=strtobool, default=True, help="tensorboardのログを消すか消さないか")
    opt = parser.parse_args()

    main(opt)