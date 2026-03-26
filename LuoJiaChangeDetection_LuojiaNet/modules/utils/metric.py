from sklearn.metrics import confusion_matrix
import numpy as np


def metrics(predictions, gts, label_values=["Unchanged", "Changed"]):
    
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))
    
    # print("Confusion matrix :")
    # print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    # print("%d pixels processed" % (total))
    # print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    # for l_id, score in enumerate(Acc):
    #     print("%s: %.4f" % (label_values[l_id], score))
    # print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    # print("F1Score :")
    # for l_id, score in enumerate(F1Score):
    #     print("%s: %.4f" % (label_values[l_id], score))
    # print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    # print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    # print("Kappa: %.4f" %(kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    # print(MIoU)
    MIoU = np.nanmean(MIoU[:5])
    # print('mean MIoU: %.4f' % (MIoU))
    # print("---")

    return MIoU










# from sklearn.metrics import confusion_matrix
# import numpy as np


# def metrics(predictions, gts, label_values=["Unchanged", "Changed"]):
#     cm = confusion_matrix(
#         gts,
#         predictions,
#         labels=range(len(label_values)))
    
#     print("Confusion matrix :")
#     print(cm)
#     # Compute global accuracy
#     total = sum(sum(cm))
#     accuracy = sum([cm[x][x] for x in range(len(cm))])
#     accuracy *= 100 / float(total)
#     print("%d pixels processed" % (total))
#     print("Total accuracy : %.2f" % (accuracy))

#     Acc = np.diag(cm) / cm.sum(axis=1)
#     for l_id, score in enumerate(Acc):
#         print("%s: %.4f" % (label_values[l_id], score))
#     print("---")

#     # Compute F1 score
#     F1Score = np.zeros(len(label_values))
#     for i in range(len(label_values)):
#         try:
#             F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
#         except:
#             # Ignore exception if there is no element in class i for test set
#             pass
#     print("F1Score :")
#     for l_id, score in enumerate(F1Score):
#         print("%s: %.4f" % (label_values[l_id], score))
#     print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
#     print("---")

#     # Compute kappa coefficient
#     total = np.sum(cm)
#     pa = np.trace(cm) / float(total)
#     pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
#     kappa = (pa - pe) / (1 - pe)
#     print("Kappa: %.4f" %(kappa))

#     # Compute MIoU coefficient
#     MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
#     print(MIoU)
#     MIoU = np.nanmean(MIoU[:5])
#     print('mean MIoU: %.4f' % (MIoU))
#     print("---")

#     return accuracy