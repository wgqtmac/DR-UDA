import torch
from torch import optim
import numpy as np
from sklearn import preprocessing, neighbors
import tools.triplets_utils as tu


def validate(src_model, tgt_model, src_data_loader, tgt_data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.

    with torch.no_grad():
        X, y = tu.extract_embeddings(src_model, src_data_loader)

        Xtest, ytest = tu.extract_embeddings(tgt_model, tgt_data_loader)

        clf = neighbors.KNeighborsClassifier(n_neighbors=2)
        clf.fit(X, y)
        y_pred = clf.predict(Xtest)

        acc = (y_pred == ytest).mean()

        for i in range(len(y_pred)):
            if ytest[i] == 1 and y_pred[i] == 1:
                TP += 1
            elif ytest[i] == 0 and y_pred[i] == 0:
                TN += 1
            elif ytest[i] == 1 and y_pred[i] == 0:
                FN += 1
            elif ytest[i] == 0 and y_pred[i] == 1:
                FP += 1

        print('TP:{}, TP+FN:{}, TN:{}, TN+FP:{}'.format(TP, TP + FN, TN, TN + FP))

        TP_rate = float(TP / (TP + FN))
        TN_rate = float(TN / (TN + FP))

        HTER = 1 - (TP_rate + TN_rate) / 2

        print('TP rate:{}, TN rate:{}, HTER:{} acc:{}'.format(float(TP / (TP + FN)), float(TN / (TN + FP)), HTER, acc))

        return acc, HTER
