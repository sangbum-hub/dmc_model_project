import argparse, os
import numpy as np 

import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from transformers import AdamW

import gluonnlp as nlp
from kobert import get_pytorch_kobert_model, get_tokenizer
from sklearn.model_selection import train_test_split

from utils import * 
from settings import * 
from bert_utils import * 


parser = argparse.ArgumentParser(description='BMC_model')

parser.add_argument('--max_len', default=128, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--hidden_size', default=768, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--dr_rate', default=0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--num_classes', default=5, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--verbose', default=True, type=bool)
parser.add_argument('--save', default=True, type=bool)

args = parser.parse_args()
args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

set_seed(args.seed)


def training(train_loader, valid_loader, model, criterion, optimizer, args):
    best_loss = float('inf')
    for epoch in range(1, args.num_epochs+1):
        losses, accs = [], []
        model.train()
        for (token_ids, valid_length, segment_ids, label) in train_loader:
            token_ids = token_ids.to(args.device)
            valid_length = valid_length.to(args.device)
            segment_ids = segment_ids.to(args.device)
            label = label.long().to(args.device)

            pred_y = model(token_ids, valid_length, segment_ids).to(args.device)

            optimizer.zero_grad()
            train_loss = criterion(pred_y, label)
            train_loss.backward()
            optimizer.step()

            train_corr = proximity_correct(pred_y, label)
            train_corr /= len(label)
            accs.append(train_corr)
            losses.append(train_loss.data.cpu().numpy())
        
        valid_loss, valid_corr = evaluating(valid_loader, model, criterion, args)
        if args.verbose:
            print(f'epoch : {epoch}/{args.num_epochs}')
            print(f'train loss : {train_loss.data.cpu().numpy():.5f}, train_acc : {train_corr*100:.3f}%')
            print(f'valid loss : {valid_loss:.5f}, valid_acc : {valid_corr*100:.3f}%')
        
        if best_loss > valid_loss:
            best_loss = valid_loss 
            best_epoch = epoch
            print( f'best loss:{valid_loss:.4f}, best epoch: {best_epoch}')
            if args.save:
                if not os.path.isdir(PARAM_PATH):
                    os.mkdir(os.path.join(PARAM_PATH))
                path = f'{PARAM_PATH}/{file_name}.pt'
                torch.save(model.state_dict(prefix=''), path)


def evaluating(loader, model, criterion, args):
    losses, accs = [], []
    model.eval()
    for (token_ids, valid_length, segment_ids, label) in loader:
        token_ids = token_ids.to(args.device)
        valid_length = valid_length.to(args.device)
        segment_ids = segment_ids.to(args.device)
        label = label.long().to(args.device)

        pred_y = model(token_ids, valid_length, segment_ids).to(args.device)
        loss = criterion(pred_y, label)

        correct = proximity_correct(pred_y, label)
        correct /= len(label)
        accs.append(correct)
        losses.append(loss.data.cpu().numpy())
    return np.mean(losses), np.mean(accs) 

if __name__ == '__main__':
    bert_models, vocab = get_pytorch_kobert_model()
    bert_tokenizer = get_tokenizer()
    tokenizer = nlp.data.BERTSPTokenizer(bert_tokenizer, vocab, lower=False)

    model = BERTClassifier(model=bert_models, dr_rate=args.dr_rate, hidden_size=args.hidden_size, num_classes=args.num_classes).to(args.device)
    optimizer = AdamW(params = model.parameters(), betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    for file in FILE_PATH:
        df = preprocessing(file)
        trainset, validset = train_test_split(df.iloc[:, 2:], test_size=0.2, random_state=args.seed)

        trainloader = get_dataloader(trainset, batch_size=args.batch_size, tokenizer=tokenizer, max_len=args.max_len)
        validloader = get_dataloader(validset, batch_size=args.batch_size, tokenizer=tokenizer, max_len=args.max_len)
        file_name = os.path.basename(file).split('.')[0]
        print(f'{file_name} Training'.center(60, '-'))
        training(trainloader, validloader, model, criterion, optimizer, args)
