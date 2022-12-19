import torch 
import torch.nn as nn
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm.notebook import tqdm
import os
from dotenv import load_dotenv
load_dotenv()

from trainer import Trainer

def main(opt):
    writer = SummaryWriter(opt.log_path)
    trainer = Trainer(opt)

    summary(trainer.network, (1, int(os.environ['NBITS'])))

    for i in tqdm(range(opt.batch_size)):
        loss = trainer.train()
        loss = loss.to('cpu').detach().numpy().copy()
        writer.add_scalar('loss/', loss)



if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", type=int, help="set your dataframe's path")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--log_path", type=str, help="log file path")
    option = parser.parse_args()

    main(option)