import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

names22 = ["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12","t13","t14","t15","t16","t17","t18","t19","t20","t21","t22",
           "m1","m2","m3","m4","m5","m6","m7","m8","m9","m10","m11","m12","m13","m14","m15","m16","m17","m18","m19","m20","m21","m22",
           "label"]
names22_ = ["t31","t32","t33","t34","t35","t36","t37","t38","t39","t40","t41","t42","t43","t44","t45","t46","t47","t48","t49","t50","t51","t52",
            "m31","m32","m33","m34","m35","m36","m37","m38","m39","m40","m41","m42","m43","m44","m45","m46","m47","m48","m49","m50","m51","m52",
            "label_n"]

class Train_Loader(Dataset):
    def __init__(self, df_train, maxlen=96, winlen=22):
        super(Train_Loader, self).__init__()
        # data number
        self.winlen = winlen #22
        # data mask
        self.maxlen = maxlen #160
        # word embeded
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        # pandas dataframe
        self.data = df_train.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        txts = list(self.data[idx][:self.winlen])
        txts_n = list(self.data[idx][2*self.winlen+1:3*self.winlen+1])
        nodes_mask = list(self.data[idx][self.winlen:2*self.winlen])
        nodes_mask_n = list(self.data[idx][3*self.winlen+1:4*self.winlen+1])
        win_label = self.data[idx][2*self.winlen]
        win_label_n = self.data[idx][4*self.winlen+1]
        # word embedding
        txts_encoded = self.tokenizer(txts, padding='max_length', truncation=True, max_length=self.maxlen, return_tensors='pt')
        txts_encoded_n = self.tokenizer(txts_n, padding='max_length', truncation=True, max_length=self.maxlen, return_tensors='pt')
        # tensor of token ids
        txts_ids = txts_encoded['input_ids'].squeeze(0)
        txts_ids_n = txts_encoded_n['input_ids'].squeeze(0)
        # binary tensor with "0" for padded values and "1" for the other values
        txts_mask = txts_encoded['attention_mask'].squeeze(0)
        txts_mask_ = txts_encoded_n['attention_mask'].squeeze(0)
        return txts_ids, txts_ids_n, txts_mask, txts_mask_, nodes_mask, nodes_mask_n, win_label, win_label_n

class Test_Loader(Dataset):
    def __init__(self, df_test, maxlen=144, winlen=22):
        super(Test_Loader, self).__init__()
        # data number
        self.winlen = winlen #22
        # data mask
        self.maxlen = maxlen #160
        # word embeded
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        # pandas dataframe
        self.data = df_test.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        txts = list(self.data[idx][:self.winlen])
        win_label = self.data[idx][2*self.winlen]
        # word embedding
        txts_encoded = self.tokenizer(txts, padding='max_length', truncation=True, max_length=self.maxlen, return_tensors='pt')
        # tensor of token ids
        txts_ids = txts_encoded['input_ids'].squeeze(0)
        # binary tensor with "0" for padded values and "1" for the other values
        txts_mask = txts_encoded['attention_mask'].squeeze(0)
        return txts_ids, txts_mask, win_label

def collate_train(data):
    """
    Build mini-batch tensors from a list of tuples.
    """
    txts_ids, txts_ids_n, txts_mask, txts_mask_n, nodes_mask, nodes_mask_n, wins_label, wins_label_n = zip(*data)
    txts_ids = torch.stack(txts_ids)
    txts_ids_n = torch.stack(txts_ids_n)
    txts_mask = torch.stack(txts_mask)
    txts_mask_n = torch.stack(txts_mask_n)
    nodes_mask = torch.tensor(nodes_mask)
    nodes_mask_n = torch.tensor(nodes_mask_n)
    wins_label = torch.tensor(wins_label)
    wins_label_n = torch.tensor(wins_label_n)
    return dict(txts_ids=txts_ids, txts_ids_n=txts_ids_n, txts_mask=txts_mask, txts_mask_n=txts_mask_n, 
                nodes_mask=nodes_mask, nodes_mask_n=nodes_mask_n, wins_label=wins_label, wins_label_n=wins_label_n)


def collate_test(data):
    txts_ids, txts_mask, labels = zip(*data)
    txts_ids = torch.stack(txts_ids)
    txts_mask = torch.stack(txts_mask)
    labels = torch.tensor(labels)
    return dict(txts_ids=txts_ids, txts_mask=txts_mask, labels=labels)
