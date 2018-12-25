#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def _extract_info(items, label_split):
    labels = []
    ordered_features = []
    unordered_features = {}
    for item in items:
        labels.append(item.get("label", "?"))
        ordered_features.append(item.get("ordered", "?"))
        for key, value in item.get("unordered", {}).items():
            if key not in unordered_features:
                unordered_features[key] = []
            unordered_features[key].append(value)
    labels = np.array(labels)
    labels = (labels > label_split).astype(int)
    return labels, ordered_features, unordered_features

def _categories_to_one_hot(labels, features):
    categories = []
    xlabels = []
    ncols = 0
    if labels is None:
        labels = [None] * len(features)
    for label, category in zip(labels, features):
        if category != "?":
            categories.append(category)
            xlabels.append(label)
            if type(category) is not list:
                ncols = max(ncols, category + 1)
            else:
                for entry in category:
                    ncols = max(ncols, entry + 1)
    bin_features = np.zeros([len(xlabels), ncols])
    for i, category in enumerate(categories):
        if type(category) is not list:
            bin_features[i, category] = 1
        else:
            for entry in category:
                bin_features[i, entry] = 1
    xlabels = np.array(xlabels)
    return xlabels, bin_features

def _categories_to_proba(features, clf):
    _, bin_features = _categories_to_one_hot(None, features)
    proba = clf.predict_proba(bin_features)[:, 0]
    proba_feature = list(features)
    j = 0
    for i in range(len(features)):
        if features[i] != "?":
            proba_feature[i] = proba[j]
            j += 1
    return proba_feature

def _unordered_regression(unordered_features, labels):
    clfs = {}
    for key, features in unordered_features.items():
        xlabels, bin_features = _categories_to_one_hot(labels, features)
        clf = LogisticRegression(random_state=0).fit(bin_features, xlabels)
        clfs[key] = clf
    return clfs

def _concat_features(ordered_features, unordered_features, clfs):
    features_to_concat = [np.array(ordered_features)]
    for key in clfs.keys():
        proba_feature = _categories_to_proba(unordered_features[key], clfs[key])
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

def _prepare_for_xgboost(items, label_split, filename):
    labels, ordered_features, unordered_features = _extract_info(items, label_split)
    clfs = _unordered_regression(unordered_features, labels)
    features = _concat_features(ordered_features, unordered_features, clfs)
    _write_to_txt(labels, features, filename)

def train_model(items, label_split):
    filename = "/tmp/ram/diabetic_train_%d.txt" % label_split
    _prepare_for_xgboost(items, label_split, filename)
    
    dtrain = xgb.DMatrix(filename)
    param = {'max_depth':7, 'objective':'binary:logistic'}
    num_round = 100
    bst = xgb.train(param, dtrain, num_round)

    return bst

def test_model(items, bst1, bst2):
    filename1 = "/tmp/ram/diabetic_test_1.txt"
    _prepare_for_xgboost(items, 1, filename1)
    dtest1 = xgb.DMatrix(filename1)
    preds1 = bst1.predict(dtest1)
    preds1 = preds1 > 0.5

    filename2 = "/tmp/ram/diabetic_test_2.txt"
    _prepare_for_xgboost(items, 2, filename2)
    dtest2 = xgb.DMatrix(filename2)
    preds2 = bst2.predict(dtest2)
    preds2 = preds2 > 0.5

    tp = [0, 0, 0]
    fp = [0, 0, 0]
    fn = [0, 0, 0]
    for i in range(len(items)):
        if preds2[i]:
            pred = 3
        elif preds1[i]:
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

if __name__ == "__main__":
    import pickle
    with open("/tmp/ram/diabetic_data.pkl", "rb") as f:
        items = pickle.load(f)
    n = len(items)
    split = int(n * 0.9)
    train_items = items[:split]
    test_items = items[split:]
    bst1 = train_model(train_items, 1)
    bst2 = train_model(train_items, 2)
    test_model(test_items, bst1, bst2)

