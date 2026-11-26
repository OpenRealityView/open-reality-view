# Contributing to Open Reality View

Thanks for your interest. The project is at an early stage, so the most useful contributions right now are dataset adapters, baseline reproductions, evaluation code and careful bug reports. For anything larger than a small fix, please open an issue first so we can agree on the approach.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pre-commit install
```

## Before opening a pull request

- `ruff check .` and `ruff format --check .` pass
- `pytest` passes; new code comes with tests
- The PR is focused and links the issue or roadmap item it addresses

## What not to commit

- Datasets, checkpoints, logs or rendered outputs (`/data`, `/outputs`, `*.pt`, `*.ckpt`, `*.safetensors` are git-ignored)
- Third-party code with an incompatible license. When adding a dependency, model or dataset, state its license in the PR; non-commercial assets must stay optional and clearly labeled
- Images or videos you are not allowed to redistribute, such as screenshots of commercial products

## Project page

The site lives in `docs/` and is deployed by `.github/workflows/pages.yml` on every push to `main` that touches `docs/`.

- Preview locally: `python -m http.server -d docs 8000`, then open <http://localhost:8000>
- Videos go to `docs/static/videos/` as H.264 MP4, at most 1280 px wide and ideally under 10 MB each (GitHub rejects files over 100 MB; keep the whole site under 1 GB):

  ```bash
  ffmpeg -i input.mp4 -vf "scale='min(1280,iw)':-2" -c:v libx264 -crf 26 -preset slow \
         -pix_fmt yuv420p -movflags +faststart -an docs/static/videos/NAME.mp4
  ffmpeg -i docs/static/videos/NAME.mp4 -frames:v 1 -q:v 3 docs/static/images/demos/NAME.jpg   # poster
  ```

  Clips that are already small H.264 files only need a remux: `ffmpeg -i in.mp4 -c copy -an -movflags +faststart docs/static/videos/NAME.mp4`.

## License of contributions

By submitting a contribution you agree that it is licensed under the Apache License 2.0, as described in Section 5 of the [license](LICENSE).
