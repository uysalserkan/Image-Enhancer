"""Image enhancer service."""
from argparse import ArgumentParser


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

    args = parser.parse_args()
