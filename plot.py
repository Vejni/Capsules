import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    files = [
        # "BaseCNN", "DynamicCaps", "VarCaps", "SRCaps"
        "BaseCNN", "NazeriCNN", "DynamicCaps", "SRCaps", "VarCaps"
    ]
    figure = plt.gcf() # get current figure
    figure.set_size_inches(12, 9)

    for f in files:
        with open("C:/Marci/Suli/Dissertation/Repository/models/outs/BACH/BACH_" + f + ".out", 'r', encoding = 'cp850') as fin:
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

        print(len(val_loss), len(val_acc), len(train_acc), len(train_loss))
        xaxis = np.arange(1, len(train_loss) + 1, 1.0)
        xmax = 10
        # Loss
        plt.subplot(2, 2, 1)
        plt.title("Training Loss")
        plt.plot(xaxis, train_loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        #plt.legend(frameon=False)
        plt.xlim([1, xmax])

        plt.subplot(2, 2, 2)
        plt.title("Validation Loss")
        plt.plot(xaxis, val_loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        #plt.legend(frameon=False)
        plt.xlim([1, xmax])

        # Accuracy
        plt.subplot(2, 2, 3)
        plt.title("Training Accuracy")
        plt.plot(xaxis, train_acc)
        plt.xlabel("Epochs")
        plt.ylabel("Acc")
        #plt.legend(frameon=False)
        plt.xlim([1, xmax])

        plt.subplot(2, 2, 4)
        plt.title("Validation Accuracy")
        plt.plot(xaxis, val_acc, label=f)
        plt.xlabel("Epochs")
        plt.ylabel("Acc")
        plt.legend(frameon=False)
        plt.xlim([1, xmax])

    #plt.show()
    #plt.xticks(np.arange(1, 17, 3.0))
    plt.savefig("C:\Marci\Suli\Dissertation\Repository\docs\plots\BACH_Imagewise.png", dpi = 100)