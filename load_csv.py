#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from IPython import embed

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
    return items

def map_categories(mapping, category):
    if category not in mapping:
        mapping[category] = len(mapping) - 1
    return mapping[category]

def handle_simple_categories(unordered, mappings, key, value):
    if key not in mappings:
        mappings[key] = {"?":"?"}
    i = map_categories(mappings[key], value)
    unordered[key] = i

def parse_item(dict_row, mappings):
    label = 0
    ordered = []
    unordered = {}
    for key, value in dict_row.items():
        if key == "encounter_id":
            pass
        elif key == "patient_nbr":
            pass
        elif key == "race":
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
        elif key in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
            try:
                i = int(value) - 1
            except:
                i = "?"
            unordered[key] = i
        elif key == "time_in_hospital":
            pass
        elif key == "payer_code":
            pass
        elif key == "medical_specialty":
            pass
        elif key == "num_lab_procedures":
            pass
        elif key == "num_procedures":
            pass
        elif key == "num_medications":
            pass
        elif key == "number_outpatient":
            pass
        elif key == "number_emergency":
            pass
        elif key == "number_inpatient":
            pass
        elif key == "diag_1":
            pass
        elif key == "diag_2":
            pass
        elif key == "diag_3":
            pass
        elif key == "number_diagnoses":
            pass
        elif key == "max_glu_serum":
            pass
        elif key == "A1Cresult":
            pass
        elif key == "metformin":
            pass
        elif key == "repaglinide":
            pass
        elif key == "nateglinide":
            pass
        elif key == "chlorpropamide":
            pass
        elif key == "glimepiride":
            pass
        elif key == "acetohexamide":
            pass
        elif key == "glipizide":
            pass
        elif key == "glyburide":
            pass
        elif key == "tolbutamide":
            pass
        elif key == "pioglitazone":
            pass
        elif key == "rosiglitazone":
            pass
        elif key == "acarbose":
            pass
        elif key == "miglitol":
            pass
        elif key == "troglitazone":
            pass
        elif key == "tolazamide":
            pass
        elif key == "examide":
            pass
        elif key == "citoglipton":
            pass
        elif key == "insulin":
            pass
        elif key == "glyburide-metformin":
            pass
        elif key == "glipizide-metformin":
            pass
        elif key == "glimepiride-pioglitazone":
            pass
        elif key == "metformin-rosiglitazone":
            pass
        elif key == "metformin-pioglitazone":
            pass
        elif key == "change":
            pass
        elif key == "diabetesMed":
            pass
        elif key == "readmitted":
            label = 1 if value == "<30" else 2 if value == ">30" else 0
        else:
            raise Exception("Unknown Feature: %s" % key)
    item = {"label":label, "ordered":ordered, "unordered":unordered}
    return item

if __name__ == "__main__":
    dict_data = load_csv("/tmp/ram/diabetic_data.csv")
    items = parse_data(dict_data)
    print(*items[:20], sep="\n")

