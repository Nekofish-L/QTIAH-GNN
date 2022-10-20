import torch
import torch.nn as nn
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score


def train(g, model, features, num_epoch, learning_rate, loss_f, device, beta, eta_min):
    """
    train function
    """
    g = g.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=eta_min)
    best_val_auc = 0

    labels = g.ndata["y"]['company'].to(device)
    train_mask = g.ndata["train_mask"]['company'].to(device)
    test_mask = g.ndata["test_mask"]['company'].to(device)
    val_mask = g.ndata["val_mask"]['company'].to(device)

    cls_num_list = [
        (g.nodes['company'].data['y'][train_mask].shape[0] - g.nodes['company'].data['y'][train_mask].sum()).item(),
        g.nodes['company'].data['y'][train_mask].sum().item()]

    for e in range(num_epoch):
        # forward
        feat = model(g, features)
        result = feat['company']

        pred = result.argmax(dim=1)

        # loss_func = loss_f(cls_num_list, max_m=0.5, s=10, weight = torch.FloatTensor([1,20]).to(device))
        loss_func = loss_f(cls_num_list, max_m=0.5, s=10,
                           weight=(1 - beta) / (1 - beta ** torch.tensor(cls_num_list).to(device)))

        pseudo_loss = sum(feat['pseudo_loss']) / len(feat['pseudo_loss'])
        loss = 2 * loss_func(result[train_mask], labels[train_mask]) + pseudo_loss
        val_recall = recall_score(labels[val_mask].cpu(), pred[val_mask].cpu())
        val_auc = roc_auc_score(labels[val_mask].cpu(), pred[val_mask].cpu())
        val_b_acc = balanced_accuracy_score(labels[val_mask].cpu(), pred[val_mask].cpu())

        if best_val_auc < val_auc:
            best_val_auc = val_auc
            pred_best = pred.cpu().detach()
            res_best = result.cpu().detach()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(
            'In epoch {}, loss: {:.3f}, val auc:{:.3f} (best:{:.3f}), recall:{:.3f}, bacc:{:.3f}'.format(e, loss.item(),
                                                                                                         val_auc,
                                                                                                         best_val_auc,
                                                                                                         val_recall,
                                                                                                         val_b_acc))

    labels = labels.cpu()

    test_auc = roc_auc_score(labels[test_mask], nn.functional.softmax(res_best[test_mask], dim=-1)[:, 1].detach())
    test_recall = recall_score(labels[test_mask], pred_best[test_mask])

    test_b_acc = balanced_accuracy_score(labels[test_mask], pred_best[test_mask])

    print('test: auc:{.3f}, recall:{:.3f}, bacc:{:.3f}'.format(test_auc, test_recall, test_b_acc))
