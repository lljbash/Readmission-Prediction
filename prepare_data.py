#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
import pickle

def load_csv(csvname):
    dict_data = []
    with open(csvname, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            dict_data.append(row)
    return dict_data

def parse_data(dict_data):
    items = []
    mappings = {}
    for dict_row in dict_data:
        item = parse_item(dict_row, mappings)
        items.append(item)
    category_count = {key:len(mappings[key]) - 1 for key in mappings.keys()}
    return items, category_count

def map_categories(mapping, category):
    if category not in mapping:
        mapping[category] = len(mapping) - 1
    return mapping[category]

def handle_simple_categories(unordered, mappings, key, value):
    if key not in mappings:
        mappings[key] = {"?":"?"}
    i = map_categories(mappings[key], value)
    unordered[key] = i

def handle_listed_categories(unordered, mappings, key, value):
    if key not in mappings:
        mappings[key] = {"?":"?"}
    i = map_categories(mappings[key], value)
    if key not in unordered:
        unordered[key] = []
    if i != "?":
        unordered[key].append(i)

def parse_item(dict_row, mappings):
    label = 0
    ordered = []
    unordered = {}
    for key, value in dict_row.items():
        if key in ["encounter_id"]:
            pass
        elif key == "patient_nbr":
            pass
        elif key in ["race", "medical_specialty", "admission_type_id", "discharge_disposition_id", "admission_source_id", "payer_code"]:
            handle_simple_categories(unordered, mappings, key, value)
        elif key == "gender":
            i = 1 if value == "Female" else 2 if value == "Male" else "?"
            ordered.append(i)
        elif key in ["age", "weight"]:
            try:
                i = int(value[1:-1].split("-")[0])
            except:
                i = "?"
            ordered.append(i)
        elif key in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]:
            try:
                i = int(value)
            except:
                i = "?"
            ordered.append(i)
        elif key in ["diag_1", "diag_2", "diag_3"]:
            handle_listed_categories(unordered, mappings, "diag", value)
        elif key in ["max_glu_serum", "A1Cresult"]:
            i = "?" if value == "None" else 0 if value == "Norm" else int(re.findall(r'\d+', value)[0])
            ordered.append(i)
        elif key in ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]:
            # medmapping = {"No":0, "Down":1, "Steady":2, "Up":3}
            # i = medmapping.get(value, "?")
            # ordered.append(i)
            handle_listed_categories(unordered, mappings, "medcine", key+"+"+value)
        elif key in ["change", "diabetesMed"]:
            i = 0 if value == "No" else 1
            ordered.append(i)
        elif key == "readmitted":
            label = 1 if value == "<30" else 2 if value == ">30" else 3
        else:
            raise Exception("Unknown Feature: %s" % key)
    item = {"label":label, "ordered":ordered, "unordered":unordered}
    return item

if __name__ == "__main__":
    dict_data = load_csv("/tmp/ram/diabetic_data.csv")
    items, category_count = parse_data(dict_data)
    # print(*items[:30], sep="\n")
    with open("/tmp/ram/diabetic_data.pkl", "wb") as f:
        pickle.dump({"items": items, "category_count": category_count}, f)

