from src.datasets import set_seed
from src.models import PatchWiseModel
from argparse import Namespace

if __name__ == "__main__":
    set_seed()

    args = Namespace(
        batch_size=32,
        lr=0.0001,
        epochs=50
    )

    patch_wise_model = PatchWiseModel(input_size=[3, 512, 512], classes=3, channels=3, output_size=[3, 32, 32])
    patch_wise_model.train("./data/Histopathological_Graded", args)