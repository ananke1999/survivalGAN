"""
Vanilla GAN inference server.

Loads the checkpoint bundle in ../gan_checkpoint/ once at startup and exposes:
    POST /api/generate   -> sample one synthetic patient

The decoding logic mirrors test_gan.py: G(z) -> inverse-scale -> round/clip
discrete columns -> clamp time >= 0.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from flask import Flask, jsonify
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "gan_checkpoint"

# model_gan.py lives in the checkpoint folder next to the weights
sys.path.insert(0, str(CKPT))
from model_gan import Generator  # noqa: E402

with open(CKPT / "metadata_gan.json") as f:
    META = json.load(f)
SCALER = joblib.load(CKPT / "scaler_gan.joblib")

G = Generator(META["latent_dim"], META["n_features"])
G.load_state_dict(torch.load(CKPT / "generator_gan.pt", map_location="cpu"))
G.eval()

print(
    f"Loaded generator: latent_dim={META['latent_dim']} "
    f"n_features={META['n_features']} "
    f"params={sum(p.numel() for p in G.parameters()):,}"
)

app = Flask(__name__)
CORS(app)


def _decode(z: torch.Tensor) -> dict:
    with torch.no_grad():
        fake_scaled = G(z).numpy()
    fake_raw = SCALER.inverse_transform(fake_scaled)[0]

    columns = META["columns"]
    values = {c: float(v) for c, v in zip(columns, fake_raw)}

    for c in META["discrete_columns"]:
        lo, hi = META["discrete_bounds"][c]
        values[c] = int(np.clip(round(values[c]), lo, hi))

    if "time" in values:
        values["time"] = max(0.0, float(values["time"]))

    return {
        "columns": columns,
        "values": values,
        "latent_preview": z[0, :8].tolist(),
        "raw_preview": fake_scaled[0, :8].tolist(),
    }


@app.post("/api/generate")
def generate():
    z = torch.randn(1, META["latent_dim"])
    return jsonify(_decode(z))


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "n_features": META["n_features"]})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
