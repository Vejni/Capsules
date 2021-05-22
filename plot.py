import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    files = [
        #"Basecnn", "Dynamic", "SRcaps"
        "Patchwise lr001", "Patchwise lr0001", "Patchwise Original lr0001"
    ]
    figure = plt.gcf() # get current figure
    figure.set_size_inches(12, 9)

    for f in files:
        with open("C:/Marci/Suli/Dissertation/Repository/models/outs/breakhis_" + f + ".out", 'r', encoding = 'cp850') as fin:
            lines = fin.readlines()
        
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        for i, line in enumerate(lines):
            if line.startswith("train Loss"):
                train_loss.append(float(line.split(" ")[2].strip()))
                train_acc.append(float(line.split(" ")[4].strip()))

            if line.startswith("val Loss"):
                val_loss.append(float(line.split(" ")[2].strip()))
                val_acc.append(float(line.split(" ")[4].strip()))

        xaxis = np.arange(1, len(train_loss) + 1, 1.0)
        xmax = 10
        # Loss
        plt.subplot(2, 2, 1)
        plt.plot(xaxis, train_loss, label=f + ' Training loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(frameon=False)
        plt.xlim([1, xmax])

        plt.subplot(2, 2, 2)
        plt.plot(xaxis, val_loss, label=f + ' Validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(frameon=False)
        plt.xlim([1, xmax])

        # Accuracy
        plt.subplot(2, 2, 3)
        plt.plot(xaxis, train_acc, label=f + ' Training Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Acc")
        plt.legend(frameon=False)
        plt.xlim([1, xmax])

        plt.subplot(2, 2, 4)
        plt.plot(xaxis, val_acc, label=f + ' Validation Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Acc")
        plt.legend(frameon=False)
        plt.xlim([1, xmax])

    #plt.show()
    #plt.xticks(np.arange(1, 17, 3.0))
    plt.savefig("C:\Marci\Suli\Dissertation\Repository\docs\plots\Bach_patchwise.png", dpi = 100)