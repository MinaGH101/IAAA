from pathlib import Path
import numpy as np


def get_sum(x, y):

    print(x+y)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--x", required=True, type=str)
    parser.add_argument("-y", "--y", required=True, type=str)

    args = parser.parse_args()

    x = args.x
    y = args.y

    get_sum(x, y)
