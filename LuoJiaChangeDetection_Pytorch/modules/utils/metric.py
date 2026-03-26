from sklearn.metrics import confusion_matrix
import numpy as np


def metrics(predictions, gts, label_values=["Unchanged", "Changed"], logger_handle=None):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))
    
    print("Confusion matrix :")
    if logger_handle:
        logger_handle.info("Confusion matrix :")
    print(cm)
    if logger_handle:
        logger_handle.info(f"Confusion matrix:\n{cm}")
    
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    if logger_handle:
        logger_handle.info("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))
    if logger_handle:
        logger_handle.info("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
        if logger_handle:
            logger_handle.info("%s: %.4f" % (label_values[l_id], score))
    print("---")
    if logger_handle:
        logger_handle.info("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    print("F1Score :")
    if logger_handle:
        logger_handle.info("F1Score :")
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
        if logger_handle:
            logger_handle.info("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    if logger_handle:
        logger_handle.info('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    print("---")
    if logger_handle:
        logger_handle.info("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" %(kappa))
    if logger_handle:
        logger_handle.info("Kappa: %.4f" % (kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    if logger_handle:
        logger_handle.info(f"{MIoU}")
    MIoU = np.nanmean(MIoU[:5])
    print('mean MIoU: %.4f' % (MIoU))
    if logger_handle:
        logger_handle.info('mean MIoU: %.4f' % (MIoU))
    print("---")
    if logger_handle:
        logger_handle.info("---")

    return accuracy
