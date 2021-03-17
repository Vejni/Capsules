from src.datasets import set_seed
from src.models import PatchWiseModel
from argparse import Namespace

if __name__ == "__main__":
    set_seed()

    args = Namespace(
        batch_size=32,
        lr=0.0001,
        epochs=5,
        augment=True,
        workers=0
    )

    patch_wise_model = PatchWiseModel(input_size=[3, 512, 512], classes=3, channels=3, output_size=[3, 64, 64])
    patch_wise_model.train("C:\Marci\Suli\Dissertation\Repository\data", args)
    patch_wise_model.plot_metrics()
    patch_wise_model.test("C:\Marci\Suli\Dissertation\Repository\data", args)
    patch_wise_model.test_separate_classes("C:\Marci\Suli\Dissertation\Repository\data", args)