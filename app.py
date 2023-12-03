"""Image enhancer API."""
from fastapi import FastAPI

from enhance_images import init_models, upscale_images


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


CONSTANTS = AttributeDict({
    "tile": 0,
    "tile_pad": 10,
    "pre_pad": 0,
    "half": False,
    "gpu_id": None,
    "outscale": 4,
    "has_aligned": False,
    "only_center_face": False,
    "paste_back": True
})

app = FastAPI(
    title="Image Enhancer API Service",
    version="1.0.0",
)


@app.post("/enhance")
def _enhance():
    pass


if __name__ == '__main__':
    model = init_models(args=CONSTANTS)
