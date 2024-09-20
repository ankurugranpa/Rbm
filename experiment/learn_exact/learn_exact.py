import sys

import csv
import matplotlib.pyplot as plt
import numpy as np


def int_to_4bit_binary(n):
    if n < 0 or n > 15:
        raise ValueError("only 0-15")
    return format(n, '04b')

args = sys.argv
def main():

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # sampling data distribution
    with open(args[1] + "/befor_p_v_distribution.csv") as f:
        reader = csv.reader(f)

        num = 0
        y = []
        for row in reader:
            y.append(np.double(row[0]))
            num += 1
        x = np.arange(num)
        y = np.array(y)

    ax1.set_xlim(0, x.size-1)
    ax1.set_xlabel("Status")
    ax1.set_ylabel("Probability")
    ax1.bar(x, y, linewidth=1.0, width=1.0, align="edge")
    ax1.set_title(r"Distribution of Generative Models; P(v|$\theta$)")

    # before learn distribution
    with open(args[1] + "/after_buf_p_v_distribution.csv") as f:
        reader = csv.reader(f)

        num = 0
        y = []
        for row in reader:
            y.append(np.double(row[0]))
            num += 1
        x = np.arange(num)
        y = np.array(y)
        # size = reader.

    ax2.set_xlim(0, x.size-1)
    ax2.set_xlabel("Status")
    ax2.set_ylabel("Probability")
    ax2.bar(x, y, linewidth=1.0, width=1.0, align="edge")
    ax2.set_title(r"Distribution of Pre-training Learning Models; P(v|$\theta$)")



    # data set histogram
    with open(args[1] + "/sampling_data.csv") as f:
        x = []
        y = []
        y_num = []
        reader = csv.reader(f)
        for row in reader:
            y.append(row[0].replace(" ", ""))

        for i in range(15):
            x.append(i)
            y_num.append((y.count(int_to_4bit_binary(i))))

        x = np.array(x)
        y_num = np.array(y_num)

    ax3.set_xlim(0, 15)
    ax3.bar(x, y_num, color="g", linewidth=1.0, width=1.0, align="edge")
    ax3.set_title(r"Histogram of Generative Models; P(v|$\theta$)")
    ax3.set_xlabel("Status")
    ax3.set_ylabel("Number")

    # learned distribution
    with open(args[1] + "/after_p_v_distribution.csv") as f:
        reader = csv.reader(f)

        num = 0
        y = []
        for row in reader:
            y.append(np.double(row[0]))
            num += 1
        x = np.arange(num)
        y = np.array(y)
        # size = reader.

    ax4.set_xlim(0, x.size-1)
    ax4.set_xlabel("Status")
    ax4.set_ylabel("Probability")
    ax4.bar(x, y, linewidth=1.0, width=1.0, align="edge")
    ax4.set_title(r"Distribution of Post-training Learning Models; P(v|$\theta$)")

    plt.tight_layout()
    plt.savefig("rbm_result_test.png", format="png", dpi=300)

if __name__=="__main__":
    main()

