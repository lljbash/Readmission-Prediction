#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import configparser
from prepare_data import prepare_data
from model import run_model

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    data_path = config["Path"]["DataPath"]
    tmpdir = config["Path"]["TempDirectory"]
    prepare_data(data_path, tmpdir)
    run_model(tmpdir)

