from src.datasets import set_seed
from src.patchwisemodel import PatchWiseModel
from src.imagewisemodels import BaseCNN
from argparse import Namespace

if __name__ == "__main__":
    set_seed()

    args_patch_wise = Namespace(
        batch_size=32,
        lr=0.0001,
        epochs=1,
        augment=True,
        workers=4,
        classes=3,
        data_path="./data/patchwise_dataset"
    )

    args_img_wise = Namespace(
        lr=0.0001,
        epochs=1,
        augment=True,
        workers=4,
        classes=3,
        data_path="./data/imagewise_dataset"
    )

    patch_wise_model = PatchWiseModel(input_size=[3, 512, 512], classes=3, channels=3, output_size=[3, 64, 64])
    #patch_wise_model.train_model(args_patch_wise)
    #patch_wise_model.plot_metrics()
    #patch_wise_model.test(args_patch_wise)
    #patch_wise_model.test_separate_classes(args_patch_wise)
    path = patch_wise_model.save_model("./models/")

    image_wise_model = BaseCNN(input_size=[3, 512, 512], classes=3, channels=3, output_size=[3, 64, 64], patchwise_path=path, args=args_img_wise)
    image_wise_model.train_model(args_img_wise)