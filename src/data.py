"""
Module to load, process and analyze input data
"""
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random


def load_csv_to_df(file_path):
    """
    CSV 파일에서 데이터프레임 타입으로 load하기 위한 함수
    Args:
        - file_path: path to file + file name
    Returns:
        - loaded dataframe
    """
    df = pd.read_csv(file_path)
    return df


def sample_df(df, sample_rate):
    """
    데이터를 소수만 sampling 하여 리턴
    Args:
        - df
        - sample_rate: sampling rate (from 0 - 1)
    Returns:
        - sampled dataframe
    """
    return df.sample(frac=sample_rate, replace=True, random_state=1)


def sample_and_export(df, sample_rate, output_file_path):
    export_to_csv(
        sample_df(df, sample_rate),
        output_file_path
    )


def get_df_labels(df, label_column):
    """
    Args:
        - df
        - label_column: column name containing labels / classes
    Returns:
        - List of unique labels / classes
    """
    labels = df[label_column].unique()
    return labels


def plot_df_class_balance(df, label_column):
    """
    Args:
        - df
        - label_column: column name containing labels / classes
    Returns:
        - Plots label / class distribution histogram and returns
            dictionary of { Class/Label: n of appearances }
    """
    plt.title("Distribution of labels")
    plt.bar(dict(Counter(df[label_column])).keys(),
            dict(Counter(df[label_column])).values())
    return Counter(df[label_column])


def export_to_csv(df, output_file_path):
    """
    This function exports a datframe to a CSV file in the same root directory
    where the .py file is being executed.

    Args:
        - df
    Returns:
        - output_file_path
    """
    df.to_csv(output_file_path, index=False)


def add_well_known_port_column(row):
    """
    Method to add a new categorical column to the df, which states if the destination
    port of the flow connection is a well-known port.

    Args:
        - row: df row
    Returns:
        - True if port is well-known, else False
    """
    well_known_ports = {21, 22, 80, 8080, 443}
    if row[" Destination Port"] in well_known_ports:
        return True
    else:
        return False


def modify_portscan_attack_behavior(df):
    """
    Method to add synthetic noise to the average packet sizes.

    Args:
        - df
    Returns:
        - df
    """
    for i, row in df.iterrows():
        if row[" Label"] == "PortScan":
            seed = random.randint(0, 1)
            if seed == 0:
                df.at[i, " Average Packet Size"] = row[" Average Packet Size"] + \
                    random.randint(0, 100)

    return df
