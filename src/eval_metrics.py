import numpy as np

try:
    from sklearn.metrics import accuracy_score, f1_score
except ModuleNotFoundError:
    def accuracy_score(y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        return np.mean(y_true == y_pred)

    def f1_score(y_true, y_pred, average="weighted"):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        labels = np.union1d(y_true, y_pred)
        f1_values = []
        supports = []
        for label in labels:
            true_label = y_true == label
            pred_label = y_pred == label
            tp = np.sum(true_label & pred_label)
            fp = np.sum(~true_label & pred_label)
            fn = np.sum(true_label & ~pred_label)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            score = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            f1_values.append(score)
            supports.append(np.sum(true_label))

        f1_values = np.asarray(f1_values, dtype=np.float64)
        supports = np.asarray(supports, dtype=np.float64)
        if average == "weighted":
            total = supports.sum()
            return np.sum(f1_values * supports) / total if total else 0.0
        if average == "macro":
            return np.mean(f1_values) if f1_values.size else 0.0
        if average is None:
            return f1_values
        raise ValueError(f"Unsupported fallback f1 average: {average}")


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_iemocap(results, truths):
    metrics = get_iemocap_metrics(results, truths)
    print("  - F1 Score: ", metrics["f1"])
    print("  - Accuracy: ", metrics["acc"])


def get_four_class_metrics(results, truths):
    test_preds = results.view(-1, 4).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_i = np.argmax(test_preds, axis=1)
    test_truth_i = test_truth
    f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
    acc = accuracy_score(test_truth_i, test_preds_i)
    return {"f1": float(f1), "acc": float(acc)}


def get_iemocap_metrics(results, truths):
    return get_four_class_metrics(results, truths)


def eval_msp_improv(results, truths):
    metrics = get_msp_improv_metrics(results, truths)
    print("  - F1 Score: ", metrics["f1"])
    print("  - Accuracy: ", metrics["acc"])


def get_msp_improv_metrics(results, truths):
    return get_four_class_metrics(results, truths)


def get_metrics(dataset, results, truths):
    if dataset == "iemocap":
        return get_iemocap_metrics(results, truths)
    if dataset == "msp-improv":
        return get_msp_improv_metrics(results, truths)
    return {}
