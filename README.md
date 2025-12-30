# TextureGan

**Super-resolution & detail enhancement for game textures**

TextureGan is a compact, high-quality super-resolution model tuned specifically for game textures and UV atlases. It performs 2× upscaling while restoring fine detail, sharpening edges, and improving color fidelity with minimal artifacts — ideal for texture packs, game asset pipelines, and digital restoration.

---

## Highlights

- ✅ 2× super-resolution tuned for game textures (1024→2048 example)
- ✅ Edge, detail and color enhancement built-in
- ✅ Seam-aware tiling inference for large atlases
- ✅ Fast GPU inference, exportable to ONNX / FP16

---

## Release — Download

Pretrained checkpoint (2×):

```
https://github.com/Mahdiqoo/TextureGan/releases/download/Upscaling/TextureGan2x.pth
```

---

## Quick start (Python)

Install requirements (recommended in a virtualenv):

```bash
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -U pip
pip install torch torchvision pillow numpy opencv-python
# optional: export / runtime
pip install onnx onnxruntime-gpu
```

Example inference script:

```python
from PIL import Image
import torch
import numpy as np
from texturegan import TextureGanModel  # adjust to your package layout

# --- config
WEIGHTS_URL = "https://github.com/Mahdiqoo/TextureGan/releases/download/Upscaling/TextureGan2x.pth"
WEIGHTS_PATH = "weights/TextureGan2x.pth"  # download or point to this path
INPUT_PATH = "examples/input_1024.png"
OUTPUT_PATH = "examples/result_2048.png"

# load input
img = Image.open(INPUT_PATH).convert("RGB")

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# instantiate model
model = TextureGanModel(upscale=2).to(device)
state = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# preprocess
x = np.array(img).astype(np.float32) / 255.0
x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(device)

# inference (wrap in tiled inference for large textures)
with torch.no_grad():
    y = model(x)

# postprocess and save
y = y.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()
out = Image.fromarray((y * 255.0).astype("uint8"))
out.save(OUTPUT_PATH)
print("Saved:", OUTPUT_PATH)
```

**Notes**
- For large textures use tiled inference (see `utils/tiling.py`) with overlap and linear blending to avoid seams.
- The model expects 3-channel RGB images in sRGB color space. If your pipeline uses linear color space (common for PBR), convert to/from linear as needed.

---

## CLI

A minimal CLI example:

```bash
python -m texturegan.cli \
  --input examples/input_1024.png \
  --output examples/result_2048.png \
  --weights weights/TextureGan2x.pth \
  --tile 512
```

Include `--tile` (tile size) and `--overlap` arguments for high-resolution atlases.

---

## Tiling & Seam Handling

To process very large atlases safely:

1. Break the image into tiles with an overlap (e.g. tile=512, overlap=32).
2. Run inference on each tile.
3. Blend overlapping regions with linear alpha or windowed blending (Hanning/feather) to avoid seam artifacts.

A helper is provided at `utils/tiling.py` in this repo. Use seam-aware cropping during training if you plan to fine-tune on atlas datasets.

---

## Model training (overview)

If you want to reproduce or fine-tune TextureGan locally:

- **Dataset**: paired HR/LR texture atlases. Create LR by bicubic downsampling HR (scale ×2) and add synthetic degradations for robustness.
- **Losses**: L1/reconstruction + perceptual (VGG) + edge loss (Sobel) + optional adversarial loss for sharper detail.
- **Optimizer**: AdamW, lr=2e-4 (example)
- **Batch size**: 8–32 depending on GPU memory
- **HR crop size**: 256×256 (for ×2 → LR 128)
- **Training length**: 200k–600k steps depending on data size

See `training/` for example configs and training scripts.

---

## Best practices for game pipelines

- Keep color-space consistent (sRGB vs linear). Convert normals or non-color maps appropriately — TextureGan is tuned for color/painted textures, not normal or roughness maps unless retrained.
- Use seam-aware tiling and overlap blending for atlases.
- Use a separate model or retrain for normal, roughness, metallic, or height maps to avoid introducing shape/color artifacts.

---

Example Result:
<img width="900" height="487" alt="download (1)" src="https://github.com/user-attachments/assets/1c5846d1-43c9-4826-851b-86f4546a0eb4" />
