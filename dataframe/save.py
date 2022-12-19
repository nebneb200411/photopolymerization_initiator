import pandas as pd
import os, sys
sys.path.append('../')
from files.extension import ExtensionChanger

def save_as_csv(df, path):
    path = ExtensionChanger(path).replacer('.csv')
    df.to_csv(path)

def save_as_excel(df, path):
    path = ExtensionChanger(path).replacer('.xlsx')
    df.to_excel(path)