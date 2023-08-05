import torch

def MSL(y_pred, batch_nodes_labels, batch_size, k, winlen, device):
    obl = torch.tensor(0.).to(device)
    sparsity = torch.tensor(0.).to(device)

    for i in range(batch_size):
        y_anomaly = y_pred[i]
        y_normal  = y_pred[i+batch_size]

        anomaly_idx = []
        normal_idx = []
        for j in range(len(batch_nodes_labels[i])):
            if j<winlen-k+1 and batch_nodes_labels[i][j] != torch.tensor(0).to(device):
                anomaly_idx.append(j)
        for j in range(len(batch_nodes_labels[i+batch_size])):
            if j<winlen-k+1 and batch_nodes_labels[i+batch_size][j] != torch.tensor(0).to(device):
                normal_idx.append(j)

        y_anomaly_nodes = y_anomaly[anomaly_idx]
        y_normal_nodes = y_normal[normal_idx]
        
        y_anomaly_max = torch.max(y_anomaly_nodes) # anomaly
        y_normal_max = torch.max(y_normal_nodes) # normal

        obl += max(0, torch.tensor(1.0)-y_anomaly_max+y_normal_max)
        sparsity += (torch.sum(y_anomaly_nodes) + torch.sum(y_normal_nodes))*0.001

    loss_average = (obl+sparsity)/batch_size
    return loss_average
