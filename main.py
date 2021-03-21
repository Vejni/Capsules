from src.datasets import set_seed
from src.patchwisemodel import PatchWiseModel
from src.imagewisemodels import BaseCNN, DynamicCapsules
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
        data_path="./data/Graded/patchwise_dataset"
    )

    args_img_wise = Namespace(
        lr=0.0001,
        epochs=1,
        augment=True,
        workers=4,
        classes=3,
        data_path="./data/Graded/imagewise_dataset",
        routings=3,
        lr_decay=0.9,
        lam_recon=0.392
    )

    patch_wise_model = PatchWiseModel(input_size=[3, 512, 512], classes=3, channels=3, output_size=[3, 64, 64])
    #patch_wise_model.train_model(args_patch_wise)
    #patch_wise_model.plot_metrics()
    patch_wise_model.test(args_patch_wise)
    patch_wise_model.test_separate_classes(args_patch_wise)
    path = patch_wise_model.save_model("./models/")

    image_wise_model = DynamicCapsules(input_size=[3, 512, 512], classes=3, channels=3, output_size=[3, 64, 64], patchwise_path=path, args=args_img_wise)
    #image_wise_model.train_model(args_img_wise)
    image_wise_model.test(args_img_wise)
    path = patch_wise_model.save_model("./models/", "Dynamic")