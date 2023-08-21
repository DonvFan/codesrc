# coding: utf-8
import argparse
import imp
from pickletools import optimize
import time
import math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import data
import model
import os
import os.path as osp

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=200, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=100,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, help='GPU device id used', default=1)
parser.add_argument('--nhid', type = int,  default=128)
parser.add_argument('--nlayers', type = int, default=3)
parser.add_argument('--lr', type = float, default=0.0001)
parser.add_argument('--model', type = str, default='TransformerEncoder')
args = parser.parse_args()

use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

writer = SummaryWriter('../run/%s'%args.model)
# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)
nvoc = len(data_loader.vocabulary)
print(nvoc)
voc = torch.IntTensor(list(data_loader.word_id.values())).to(device).unsqueeze(dim = 0)
word_voc = data_loader.vocabulary
ninput = nhid = args.nhid

if args.model == 'RNN':
    LMModel = model.LMModel(nvoc, ninput, nhid, args.nlayers, sql_len=args.max_sql).to(device)
elif args.model == 'TransformerEncoder':
    LMModel = model.TF_OnlyEncoder(args.max_sql, nvoc).to(device)
    LMModel.encoder.layers[-1]['mha'].register_forward_hook(model.AttnHook)
elif args.model == 'Transformer':
    LMModel = model.Transformer(args.max_sql, nvoc).to(device)
else:
    raise NotImplementedError()


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(LMModel.parameters(), lr =args.lr)



def evaluate():
    end_flag = False
    data_loader.set_valid()
    LMModel.eval()
    loss_sum = torch.zeros(1).to(device)
    cnt = 0
    right_cnt = 0
    all_cnt = 0
    with torch.no_grad():
        while(not end_flag):
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            if args.model == 'RNN' or args.model == 'TransformerEncoder':
                pred = LMModel(data)
            else:
                pred = LMModel(voc, data)

            pred = torch.flatten(pred, 0, 1)
            loss = criterion(pred, target)
            writer.add_scalar('eval/loss', scalar_value=loss, global_step=(epoch-1) * args.max_sql + cnt)
            print('[epoch %d : iter %d] eval loss = %.4f'%(epoch, cnt, loss.item()))
            cnt += 1
            loss_sum += loss
            pred_voc = torch.max(pred, -1)[1]
            right_cnt += torch.sum(pred_voc == target).item()
            all_cnt += target.shape[0]

    m_acc = right_cnt / all_cnt
    perplexity = torch.exp(loss_sum/cnt)
    writer.add_scalar('eval/m_acc', scalar_value=m_acc, global_step = epoch - 1)
    writer.add_scalar('eval/PP', perplexity.item(), epoch-1)
    print('[epoch %d] eval PP = %.4f m_acc = %.4f'%(epoch, perplexity.item(), m_acc))
    return m_acc, perplexity

# Train Function
def train():
    end_flag = False
    data_loader.set_train()
    LMModel.train()
    cnt = 0
    right_cnt = 0
    all_cnt = 0
    loss_sum = torch.zeros(1).to(device)
    # torch.autograd.set_detect_anomaly(True)
    while(not end_flag):
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)
        if args.model == 'RNN' or args.model == 'TransformerEncoder':
            pred = LMModel(data)
        else:
            pred = LMModel(voc, data)

        pred = torch.flatten(pred, 0, 1)
        loss = criterion(pred, target)
        print('[epoch %d : iter %d] train loss = %.4f'%(epoch, cnt, loss.item()))
        writer.add_scalar('train/loss', scalar_value=loss, global_step = (epoch - 1) * args.max_sql + cnt)
        cnt += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        pred_voc = torch.max(pred, -1)[1]
        right_cnt += torch.sum(pred_voc == target).item()
        all_cnt += target.shape[0]
        loss_sum += loss

    m_acc = right_cnt / all_cnt
    perplexity = torch.exp(loss_sum/cnt)
    writer.add_scalar('train/m_acc', m_acc,  epoch - 1)
    writer.add_scalar('train/PP', perplexity.item(), epoch-1)
    print('[epoch %d] train PP = %.4f m_acc = %.4f'%(epoch, perplexity.item(), m_acc))

best_acc = 0.
for epoch in range(1, args.epochs+1):
    train()
    torch.cuda.empty_cache()
    acc, pp = evaluate()
    torch.cuda.empty_cache()
    torch.save({
        'm_acc': acc,
        'pp':pp,
        'state_dict': LMModel.state_dict()
        }, os.path.join('..', 'model', args.model, 'last.pth'))
    if acc > best_acc:
        best_acc = acc
        torch.save({
        'm_acc': acc,
        'pp':pp,
        'state_dict': LMModel.state_dict()
        }, os.path.join('..', 'model', args.model, 'best.pth'))
        print('Best m_acc updated to %.4f!'%(best_acc))







 