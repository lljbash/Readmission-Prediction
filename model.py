#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def _extract_info(items, label_filter):
    labels = []
    ordered_features = []
    unordered_features = {}
    for item in items:
        label = item.get("label", "?")
        label = label_filter(label)
        if label not in (0, 1):
            continue
        labels.append(label)
        ordered_features.append(item.get("ordered", "?"))
        for key, value in item.get("unordered", {}).items():
            if key not in unordered_features:
                unordered_features[key] = []
            unordered_features[key].append(value)
    return labels, ordered_features, unordered_features

def _categories_to_one_hot(labels, features, ncols):
    categories = []
    xlabels = []
    if labels is None:
        labels = [None] * len(features)
    for label, category in zip(labels, features):
        if category != "?":
            categories.append(category)
            xlabels.append(label)
    bin_features = np.zeros([len(xlabels), ncols])
    for i, category in enumerate(categories):
        if type(category) is not list:
            bin_features[i, category] = 1
        else:
            for entry in category:
                bin_features[i, entry] = 1
    xlabels = np.array(xlabels)
    return xlabels, bin_features

def _categories_to_proba(features, ncols, clf):
    _, bin_features = _categories_to_one_hot(None, features, ncols)
    proba = clf.predict_proba(bin_features)[:, 0]
    proba_feature = list(features)
    j = 0
    for i in range(len(features)):
        if features[i] != "?":
            proba_feature[i] = proba[j]
            j += 1
    return proba_feature

def _unordered_regression(unordered_features, category_count, labels):
    clfs = {}
    for key, features in unordered_features.items():
        xlabels, bin_features = _categories_to_one_hot(labels, features, category_count[key])
        clf = LogisticRegression(random_state=0, class_weight="balanced").fit(bin_features, xlabels)
        clfs[key] = clf
    return clfs

def _concat_features(ordered_features, unordered_features, category_count, clfs):
    features_to_concat = [np.array(ordered_features)]
    for key in clfs.keys():
        proba_feature = _categories_to_proba(unordered_features[key], category_count[key], clfs[key])
        proba_feature = np.array(proba_feature).reshape([-1, 1])
        features_to_concat.append(proba_feature)
    features = np.concatenate(features_to_concat, axis=1)
    return features

def _write_to_txt(labels, features, filename):
    with open(filename, "w") as f:
        for label, feature in zip(labels, features):
            line = str(label)
            for i, entry in enumerate(feature):
                if entry != "?":
                    line += " %d:%.15f" % (i, float(entry))
            print(line, file=f)

def train_model(items, category_count, label_filter):
    filename = "/tmp/ram/diabetic_train_%d_%d_%d.txt" % (label_filter(1), label_filter(2), label_filter(3))
    labels, ordered_features, unordered_features = _extract_info(items, label_filter)
    regs = _unordered_regression(unordered_features, category_count, labels)
    features = _concat_features(ordered_features, unordered_features, category_count, regs)
    _write_to_txt(labels, features, filename)
    
    dtrain = xgb.DMatrix(filename)
    label = dtrain.get_label()
    param = {'max_depth':6, 'objective':'binary:logistic', 'silent':1}
    ratio = np.sum(label == 0) / np.sum(label == 1)
    # param['scale_pos_weight'] = ratio
    num_round = 50
    bst = xgb.train(param, dtrain, num_round)

    return {"regs": regs, "bst": bst}

def test_model(items, category_count, clfs):
    preds = []
    for i, clf in enumerate(clfs):
        filename = "/tmp/ram/diabetic_test_%d.txt" % i
        labels, ordered_features, unordered_features = _extract_info(items, lambda a: 0)
        regs = clf["regs"]
        features = _concat_features(ordered_features, unordered_features, category_count, regs)
        _write_to_txt(labels, features, filename)
        dtest = xgb.DMatrix(filename)
        preds.append(clf["bst"].predict(dtest))

    tp = [0, 0, 0]
    fp = [0, 0, 0]
    fn = [0, 0, 0]
    for i in range(len(items)):
        # if preds[2][i] > preds[1][i]:
            # if preds[2][i] > preds[0][i]:
                # pred = 3
            # else:
                # pred = 1
        # else:
            # if preds[1][i] > preds[0][i]:
                # pred = 2
            # else:
                # pred = 1
        if preds[0][i] > 0.5:
            pred = 3
        elif preds[1][i] > 0.5:
            pred = 2
        else:
            pred = 1

        if pred == items[i]["label"]:
            tp[pred-1] += 1
        else:
            fp[pred-1] += 1
            fn[items[i]["label"]-1] += 1
    pre = [tp[i] / (tp[i] + fp[i]) for i in range(3)]
    rec = [tp[i] / (tp[i] + fn[i]) for i in range(3)]
    prem = np.mean(pre)
    recm = np.mean(rec)
    f1m = 2 * prem * recm / (prem + recm)
    print("pre =", pre)
    print("rec =", rec)
    print("prem =", prem)
    print("recm =", recm)
    print("f1m =", f1m)
    return f1m

if __name__ == "__main__":
    import pickle
    import random
    with open("/tmp/ram/diabetic_data.pkl", "rb") as f:
        data = pickle.load(f)
        items = data["items"]
        category_count = data["category_count"]
        random.Random(0).shuffle(items)

    n = len(items)
    nfold = 10
    fold_size = n // 10
    f1ms = []
    for i in range(10):
        train_items = items[:i*fold_size] + items[(i+1)*fold_size:]
        test_items = items[i*fold_size:(i+1)*fold_size] 

        clf1 = train_model(train_items, category_count, lambda a: 1 if a == 3 else 0)
        clf2 = train_model(train_items, category_count, lambda a: 1 if a == 2 else 0 if a == 1 else -1)
        # clf1 = train_model(train_items, category_count, lambda a: 1 if a == 1 else 0)
        # clf2 = train_model(train_items, category_count, lambda a: 1 if a == 2 else 0)
        # clf3 = train_model(train_items, category_count, lambda a: 1 if a == 3 else 0)
        # test_model(test_items, category_count, (clf1, clf2, clf3))

        f1m = test_model(test_items, category_count, (clf1, clf2))
        f1ms.append(f1m)
    f1mm = np.mean(f1ms)
    print("f1mm = ", f1mm)

