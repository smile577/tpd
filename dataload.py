import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

names22 = ["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12","t13","t14","t15","t16","t17","t18","t19","t20","t21","t22",
           "l1","l2","l3","l4","l5","l6","l7","l8","l9","l10","l11","l12","l13","l14","l15","l16","l17","l18","l19","l20","l21","l22",
           "label","phase","project"]
names22_ = ["t31","t32","t33","t34","t35","t36","t37","t38","t39","t40","t41","t42","t43","t44","t45","t46","t47","t48","t49","t50","t51","t52",
            "l31","l32","l33","l34","l35","l36","l37","l38","l39","l40","l41","l42","l43","l44","l45","l46","l47","l48","l49","l50","l51","l52",
            "label_n","phase_n","project_n"]

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
        txts_n = list(self.data[idx][2*self.winlen+3:3*self.winlen+3])
        nodes_label = list(self.data[idx][self.winlen:2*self.winlen])
        nodes_label_n = list(self.data[idx][3*self.winlen+3:4*self.winlen+3])
        win_label = self.data[idx][2*self.winlen]
        win_label_n = self.data[idx][4*self.winlen+3]
        # word embedding
        txts_encoded = self.tokenizer(txts, padding='max_length', truncation=True, max_length=self.maxlen, return_tensors='pt')
        txts_encoded_n = self.tokenizer(txts_n, padding='max_length', truncation=True, max_length=self.maxlen, return_tensors='pt')
        # tensor of token ids
        txts_ids = txts_encoded['input_ids'].squeeze(0)
        txts_ids_n = txts_encoded_n['input_ids'].squeeze(0)
        # binary tensor with "0" for padded values and "1" for the other values
        txts_mask = txts_encoded['attention_mask'].squeeze(0)
        txts_mask_ = txts_encoded_n['attention_mask'].squeeze(0)
        return txts_ids, txts_ids_n, txts_mask, txts_mask_, nodes_label, nodes_label_n, win_label, win_label_n

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
    txts_ids, txts_ids_n, txts_mask, txts_mask_n, nodes_labels, nodes_labels_n, wins_label, wins_label_n = zip(*data)
    txts_ids = torch.stack(txts_ids)
    txts_ids_n = torch.stack(txts_ids_n)
    txts_mask = torch.stack(txts_mask)
    txts_mask_n = torch.stack(txts_mask_n)
    nodes_labels = torch.tensor(nodes_labels)
    nodes_labels_n = torch.tensor(nodes_labels_n)
    wins_label = torch.tensor(wins_label)
    wins_label_n = torch.tensor(wins_label_n)
    return dict(txts_ids=txts_ids, txts_ids_n=txts_ids_n, txts_mask=txts_mask, txts_mask_n=txts_mask_n, 
                nodes_labels=nodes_labels, nodes_labels_n=nodes_labels_n, wins_label=wins_label, wins_label_n=wins_label_n)

def collate_test(data):
    txts_ids, txts_mask, labels = zip(*data)
    txts_ids = torch.stack(txts_ids)
    txts_mask = torch.stack(txts_mask)
    labels = torch.tensor(labels)
    return dict(txts_ids=txts_ids, txts_mask=txts_mask, labels=labels)
