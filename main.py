from tqdm import tqdm
from sklearn import metrics
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from model import MSLNet
from loss import MSL
from dataload import Train_Loader, Test_Loader, collate_train, collate_test
from transformers import AdamW, get_linear_schedule_with_warmup

torch.set_printoptions(profile="full")

#print(torch.__version__)
names22 = ["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12","t13","t14","t15","t16","t17","t18","t19","t20","t21","t22",
           "l1","l2","l3","l4","l5","l6","l7","l8","l9","l10","l11","l12","l13","l14","l15","l16","l17","l18","l19","l20","l21","l22",
           "label","phase","project"]
names22_ = ["t31","t32","t33","t34","t35","t36","t37","t38","t39","t40","t41","t42","t43","t44","t45","t46","t47","t48","t49","t50","t51","t52",
            "l31","l32","l33","l34","l35","l36","l37","l38","l39","l40","l41","l42","l43","l44","l45","l46","l47","l48","l49","l50","l51","l52",
            "label_n","phase_n","project_n"]

batch_size = 8
winlen = 22
maxlen = 96

def set_seed(seed, use_cuda=True):
    #random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

def train(epoch, k, model, optimizer, scheduler, criterion_msl, criterion_bce, train_loader):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    average_msl_loss = 0
    average_bce_loss = 0
    for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader)):
        wins_nodes_ids = torch.cat([batch["txts_ids"], batch["txts_ids_n"]], dim=0).to(device)
        wins_nodes_mask = torch.cat([batch["txts_mask"], batch["txts_mask_n"]], dim=0).to(device)
        wins_nodes_labels = torch.cat([batch["nodes_labels"], batch["nodes_labels_n"]], dim=0).to(device)
        wins_label = torch.cat([batch["wins_label"], batch["wins_label_n"]], dim=0).to(device)

        wins_nodes_pred, wins_pred = model(wins_nodes_ids, wins_nodes_mask, device)

        msl_loss = criterion_msl(wins_nodes_pred, wins_nodes_labels, batch_size, k, winlen, device)
        average_msl_loss += msl_loss
        bce_loss = criterion_bce(wins_pred, wins_label.float())
        average_bce_loss += loss_prop*bce_loss
        loss = msl_loss + loss_prop*bce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print('loss = ', train_loss/len(train_loader))
    scheduler.step()
    return train_loss/len(train_loader)

def val_abnormal(epoch, best_f1_score, loss, model, val_loader):
    model.eval()
    labels = []
    preds = []
    model_filename = ""
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader), desc="Testing", total=len(val_loader)):
            wins_nodes_ids = batch["txts_ids"].to(device)
            wins_nodes_mask = batch["txts_mask"].to(device)

            wins_nodes_pred, wins_pred = model(wins_nodes_ids, wins_nodes_mask, device)

            pred_int = []
            for i in range(wins_nodes_pred.shape[0]):
                fi_max = torch.max(wins_nodes_pred[i])
                if fi_max*wins_pred[i] >= 0.5:
                    pred_int.append(1.0)
                else:
                    pred_int.append(0.0)
            preds.append(torch.tensor(pred_int))

            wins_labels = batch["labels"].float().cpu().detach()
            labels.append(wins_labels)

        labels = torch.cat(labels).tolist()
        preds = torch.cat(preds).tolist()
        #print("labels")
        #print(labels)
        #print("preds")
        #print(preds)
        precision=metrics.precision_score(labels, preds, average='binary')
        recall=metrics.recall_score(labels, preds, average='binary')
        f1_score=metrics.f1_score(labels, preds, average='binary')
        print("\tprecision\t\trecall\t\tf1-score")
        print("\t{:.3f}\t\t\t{:.3f} \t\t{:.3f}".format(precision,recall,f1_score))
        #print(metrics.classification_report(labels, preds))

        if f1_score > best_f1_score:
            checkpoint = {"model": model.state_dict(), "epoch": epoch}
            model_filename = "./model/MSL_epoch"+str(epoch)+"_f1_"+str(f1_score)[:5]+"_loss_"+str(float(loss))[:5]+".ckpt"
            torch.save(checkpoint, model_filename)

    return f1_score, model_filename

def test_abnormal(model, test_loader):
    model.eval()
    labels = []
    preds = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader)):
            wins_nodes_ids = batch["txts_ids"].to(device)
            wins_nodes_mask = batch["txts_mask"].to(device)

            wins_nodes_pred, wins_pred = model(wins_nodes_ids, wins_nodes_mask, device)

            pred_int = []
            for i in range(wins_nodes_pred.shape[0]):
                fi_max = torch.max(wins_nodes_pred[i])
                if fi_max*wins_pred[i] >= 0.5:
                    pred_int.append(1.0)
                else:
                    pred_int.append(0.0)
            preds.append(torch.tensor(pred_int))

            wins_labels = batch["labels"].float().cpu().detach()
            labels.append(wins_labels)

        labels = torch.cat(labels).tolist()
        preds = torch.cat(preds).tolist()
        #print("labels")
        #print(labels)
        #print("preds")
        #print(preds)
        precision=metrics.precision_score(labels, preds, average='binary')
        recall=metrics.recall_score(labels, preds, average='binary')
        f1_score=metrics.f1_score(labels, preds, average='binary')
        print("\tprecision\t\trecall\t\tf1-score")
        print("\t{:.3f}\t\t\t{:.3f} \t\t{:.3f}".format(precision,recall,f1_score))
        #print(metrics.classification_report(labels, preds))

    return 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

k = 2
layer = 1
lr = 8e-4
weight_decay = 2e-5
loss_prop = 0.4
seed = 3407
eps = 16
is_train = 1
if __name__ == '__main__':
    set_seed(seed)

    df_train = pd.read_csv("./dataset/train.csv", names=names22+names22_, header=0, index_col=False)
    train_dataset = Train_Loader(df_train, maxlen=maxlen, winlen=winlen)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train, drop_last=True)

    df_val = pd.read_csv("./dataset/val.csv", names=names22, header=0, index_col=False)
    val_dataset = Test_Loader(df_val, maxlen=maxlen, winlen=winlen)
    val_loader = DataLoader(val_dataset, batch_size=2*batch_size, shuffle=False, collate_fn=collate_test, drop_last=True)

    model = MSLNet(layer, winlen, k).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    t_total = len(train_loader)* eps  # Necessary to take into account Gradient accumulation
    warm_up_ratio = 0.1 # 定义要预热的step 
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warm_up_ratio*t_total, num_training_steps=t_total)
    
    criterion_msl = MSL
    criterion_bce = nn.BCELoss()
    best_f1_score = 0
    best_model_filename = ""
    for epoch in tqdm(range(eps), desc="Epoch"):
        loss = train(epoch, k, model, optimizer, lr_scheduler, criterion_msl, criterion_bce, train_loader)
        f1_score, model_filename = val_abnormal(epoch, best_f1_score, loss, model, val_loader)
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_model_filename = model_filename

    # test
    print("========================================================")
    df_test = pd.read_csv("./dataset/test.csv", names=names22, header=0, index_col=False)
    test_dataset = Test_Loader(df_test, maxlen=maxlen, winlen=winlen)
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, shuffle=False, collate_fn=collate_test, drop_last=True)
    checkpoint = torch.load(best_model_filename)
    model = MSLNet(layer, winlen, k)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    test_abnormal(model, test_loader)
