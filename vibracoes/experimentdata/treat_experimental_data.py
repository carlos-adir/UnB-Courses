import numpy as np
from matplotlib import pyplot as plt


def readlines(filename: str) -> np.ndarray:
    with open(filename, "r") as file:
        alllines = file.readlines()
    alllines.pop(0)
    alllines.pop(-1)
    for i, line in enumerate(alllines):
        alllines[i] = line.replace("\n", "").split("\t")
        for j, val in enumerate(alllines[i]):
            alllines[i][j] = float(val)
    return np.array(alllines)


if __name__ == "__main__":
    folder = "massa1/"
    teste = "test1-1/"
    filename = "frqmassa1.txt"
    filename = "mfcmassa1.txt"
    filename = "tpsmassa1.txt"
    data = readlines(folder + teste + filename)

    plt.plot(data[:, 0], data[:, 1], color="r", label="1")
    plt.plot(data[:, 0], data[:, 2], color="b", label="2")
    plt.plot(data[:, 0], data[:, 3], color="g", label="3")
    plt.legend()
    plt.xlabel("Tempo $t$ (s)")
    plt.show()
    