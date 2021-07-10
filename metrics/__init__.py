from metrics.sequence_labeling import *


def batch_computeF1(labels, preds, seq_lengths, label_set):
    y_true = []
    y_pred = []
    for i in range(len(labels)):
        label = labels[i]
        pred = preds[i]
        true_len = seq_lengths[i].item()
        pred = pred[:true_len, :true_len]
        label = label[:true_len, :true_len]
        predict_entity, label_entity = get_entities(pred, label, label_set)
        y_true.append(label_entity)
        y_pred.append(predict_entity)
    
    return precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred), classification_report(y_true,y_pred,digits=4)


def get_entities(input_tensor, label, label_set):

    input_tensor, cate_pred = input_tensor.max(dim=-1)
    predict_entity = get_entity(cate_pred, label_set)
    label_entity = get_entity(label, label_set)
    return predict_entity, label_entity


def get_entity(input_tensor,label_set):
    entity = []
    for i in range(len(input_tensor)):
        for j in range(i, len(input_tensor)):
            if input_tensor[i][j] > 0:
                tmp = (label_set[input_tensor[i][j].item()], i, j)
                entity.append(tmp)
    return entity

