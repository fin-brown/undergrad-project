#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:30:01 2017

@author: finbrown1
"""

import pandas as pd

# define the column names we want and read the data from csv to a panda df

names = ("ID", "Sex", "DoB", "Event", "Distance", "Time", "Date")

data = pd.read_csv("events_best3.csv", header=None, names=names)






