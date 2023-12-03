"""Image enhancer service."""
import os
import glob
from argparse import ArgumentParser

import torch
import cv2

from real_esrgan.enhance import init_models


def select_device() -> torch.DeviceObjType:
    """Find the GPU is exist or not.

    :returns
        `torch.DeviceObjType`: Check gpu is aviable or not.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def upscale_images(model, image_pats, args) -> None:
    """Enhance all images and save it to output folder.

    :parameters
        `model`: xyz
        `image_pats`: xyz
        `args`: xyz
    """
    for path in image_pats:
        imgname, extension = os.path.splitext(os.path.basename(path))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        _, _, output = model.enhance(
            [img],
            args.has_aligned,
            args.only_center_face,
            args.paste_back
        )

        save_path = os.path.join(args.output_path, f'{imgname}{extension}')
        cv2.imwrite(save_path, output)
        print(f"{save_path} image saved.")


if __name__ == '__main__':
    parser = ArgumentParser(
        prog="Image enhancer service",
        description="Description.",
        add_help=True
    )

    parser.add_argument(
        "-i", "--input_path",
        type=str,
        required=True,
        help="Enter your images folder path."
    )

    parser.add_argument(
        "-o", "--output_path",
        type=str,
        required=True,
        help="Enter your images output folder path."
    )

    device = select_device()

    args = parser.parse_args()
    args.outscale = 4
    args.pre_pad = 0
    args.tile = 0
    args.tile_pad = 10
    args.half = False
    args.gpu = None if device == "cpu" else 0
    args.has_aligned = False
    args.only_center_face = False
    args.paste_back = True

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # Model params
    model = init_models(args=args)

    img_paths = sorted(glob.glob(os.path.join(args.input_path, '*')))

    print("Image Paths".center(50).replace(' ', '-'))
    print(img_paths)

    upscale_images(model=model, image_pats=img_paths, args=args)
    print("Upscaling done".center(50).replace(' ', '-'))

