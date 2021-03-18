from src.datasets import set_seed
from src.patchwisemodel import PatchWiseModel
from src.imagewisemodels import BaseCNN
from argparse import Namespace

if __name__ == "__main__":
    set_seed()

    args = Namespace(
        batch_size=32,
        lr=0.0001,
        epochs=1,
        augment=True,
        workers=0,
        classes=3,
        data_path="./data"
    )

    patch_wise_model = PatchWiseModel(input_size=[3, 512, 512], classes=3, channels=3, output_size=[3, 64, 64])
    patch_wise_model.train_model(args)
    patch_wise_model.plot_metrics()
    patch_wise_model.test(args)
    patch_wise_model.test_separate_classes(args)
    path = patch_wise_model.save_model("./models/")

    image_wise_model = BaseCNN(input_size=[3, 512, 512], classes=3, channels=3, output_size=[3, 64, 64], patchwise_path=path)
    image_wise_model.set_linear_layer(args)