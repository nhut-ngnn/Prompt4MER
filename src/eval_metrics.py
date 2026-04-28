import numpy as np
from sklearn.metrics import accuracy_score, f1_score


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


def eval_mosei_senti(results, truths, exclude_zero=False):
    metrics = get_mosei_metrics(results, truths, exclude_zero)
    print("MAE: ", metrics["mae"])
    print("Correlation Coefficient: ", metrics["corr"])
    print("mult_acc_7: ", metrics["mult_acc_7"])
    print("mult_acc_5: ", metrics["mult_acc_5"])
    print("F1 score: ", metrics["f1"])
    print("Accuracy: ", metrics["acc"])


def get_mosei_metrics(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    return {
        "mae": float(mae),
        "corr": float(corr),
        "mult_acc_7": float(mult_a7),
        "mult_acc_5": float(mult_a5),
        "f1": float(f_score),
        "acc": float(accuracy_score(binary_truth, binary_preds)),
    }


def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_iemocap(results, truths):
    metrics = get_iemocap_metrics(results, truths)
    print("  - F1 Score: ", metrics["f1"])
    print("  - Accuracy: ", metrics["acc"])


def get_iemocap_metrics(results, truths):
    test_preds = results.view(-1, 4).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_i = np.argmax(test_preds, axis=1)
    test_truth_i = test_truth
    f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
    acc = accuracy_score(test_truth_i, test_preds_i)
    return {"f1": float(f1), "acc": float(acc)}


def eval_meld(results, truths):
    metrics = get_meld_metrics(results, truths)
    print("  - F1 Score: ", metrics["f1"])
    print("  - Accuracy: ", metrics["acc"])


def get_meld_metrics(results, truths):
    test_preds = results.view(-1, 7).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_i = np.argmax(test_preds, axis=1)
    test_truth_i = test_truth
    f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
    acc = accuracy_score(test_truth_i, test_preds_i)
    return {"f1": float(f1), "acc": float(acc)}


def eval_sims(results, truths, exclude_zero=False):
    metrics = get_sims_metrics(results, truths, exclude_zero)
    print("MAE: ", metrics["mae"])
    print("Correlation Coefficient: ", metrics["corr"])
    print("mult_acc_5: ", metrics["mult_acc_5"])
    print("F1 score: ", metrics["f1"])
    print("Accuracy: ", metrics["acc"])


def get_sims_metrics(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a5 = np.clip(test_preds, a_min=-1., a_max=1.)
    test_truth_a5 = np.clip(test_truth, a_min=-1., a_max=1.)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    acc2 = accuracy_score(binary_truth, binary_preds)
    
    return {
        "mae": float(mae),
        "corr": float(corr),
        "mult_acc_5": float(mult_a5),
        "f1": float(f_score),
        "acc": float(acc2),
    }


def get_metrics(dataset, results, truths):
    if dataset == "mosei":
        return get_mosei_metrics(results, truths, True)
    if dataset == "mosi":
        return get_mosei_metrics(results, truths, True)
    if dataset == "iemocap":
        return get_iemocap_metrics(results, truths)
    if dataset == "meld":
        return get_meld_metrics(results, truths)
    if dataset == "sims":
        return get_sims_metrics(results, truths)
    return {}

