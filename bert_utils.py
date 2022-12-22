import gluonnlp as nlp
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn 
import torch 
import numpy as np 
import pandas as pd 
import glob 

from utils import * 
from settings import *

class BERTDataset(Dataset):
    def __init__(self, X:pd.Series, Y:pd.Series, tokenizer, max_len, pad=True, pair=False):
        '''
        Parameters
        ----------
        X: sentences
        Y: label 


        df : dataframe (trainset or testset)
        X : sentences -> str
        Y : label -> int
        '''
        self.tokenizer = tokenizer 
        self.max_len = max_len 
        self.pad = pad 
        self.pair = pair

        self.transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=self.max_len, pad=self.pad, pair=self.pair)

        self.X = X # sentence
        self.Y = Y # label
        self.sentences = [self.transform(x) for x in self.X]
        self.labels = [np.int8(y) for y in self.Y]

    def __getitem__(self, i):
        return self.sentences[i] + (self.labels[i], )

    def __len__(self):
        return len(self.labels)


class BERTClassifier(nn.Module):
    def __init__(self, model, hidden_size, num_classes, dr_rate):
        super(BERTClassifier, self).__init__()
        self.model = model 
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        if self.dr_rate: 
            self.dropout = nn.Dropout(p=self.dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):

        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.model(input_ids=token_ids, token_type_ids=segment_ids, attention_mask=attention_mask.float())
        
        if self.dr_rate: 
            pooler = self.dropout(pooler)

        output = self.classifier(pooler)
        return output



def get_file_info(get_file_name = False):
    '''
    Channels.csv
    CostStructure.csv
    CustomerRelationships.csv
    CustomerSegmentations.csv
    KeyActivities.csv
    KeyPartners.csv
    KeyResources.csv 
    RevenueStreams.csv
    ValuePropositions.csv
    '''
    file_path = glob.glob(f'{BASE_DIR}/data/*.csv') # concatenate dataset (it, manufacture)
    return sorted(file_path)

def label_encoder(data, reverse=False):
    label_dic = {
        'A' : 0,
        'B' : 1, 
        'C' : 2
    }
    if reverse:
        label_dic = {value: key for key, value in label_dic.items()}
    return label_dic[data]



def preprocessing(file_path):
    # label encoding [ e.g. A -> 0, B -> 1]
    df = pd.read_csv(file_path)
    
    df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: label_encoder(x) if x in ['A', 'B', 'C'] else np.NaN)
    df.dropna(inplace=True, axis=0)

    return df

def get_dataloader(df, batch_size, tokenizer, max_len):
    train = BERTDataset(df.iloc[:, -2], df.iloc[:, -1], tokenizer=tokenizer, max_len=max_len)
    train_loader = DataLoader(train, batch_size, shuffle=False, drop_last=False)
    return train_loader