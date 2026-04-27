# Vanilla GAN inference backend

Loads `gan_checkpoint/` once and serves the generator over HTTP for the
Angular frontend.

## Run

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Server listens on `http://127.0.0.1:5000`.

The Angular dev server (`frontend/`) is configured via `proxy.conf.json` to
forward `/api/*` to this server, so the frontend can call `/api/generate`
directly without CORS configuration.

## Endpoints

- `GET  /api/health` – returns `{ok, n_features}`
- `POST /api/generate` – samples a fresh latent vector, runs the generator,
  and returns:
  ```json
  {
    "columns": [...],
    "values": {"Age at Diagnosis": 62, ...},
    "latent_preview": [...first 8 z values...],
    "raw_preview":    [...first 8 G(z) values, pre-inverse-scale...]
  }
  ```
