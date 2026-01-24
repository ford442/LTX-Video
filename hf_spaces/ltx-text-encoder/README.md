# LTX-Video Text Encoder Space

This Gradio app provides a standalone T5 text encoder for LTX-Video. It allows you to pre-compute text embeddings and use them with the LTX-Video pipeline, skipping the text encoder loading step.

## Features

- Encodes text prompts using the T5 encoder (same as used in LTX-Video)
- Supports negative prompts
- Saves embeddings as `.pt` files
- Reduces VRAM usage during video generation by pre-computing embeddings

## Usage

1. Enter your prompt
2. Optionally enter a negative prompt
3. Click "Encode Prompt"
4. Download the generated `.pt` file
5. Use it with LTX-Video:

```bash
python inference.py \
    --embeddings_path your_embeddings.pt \
    --height 704 --width 1216 --num_frames 121 \
    --seed 42 --pipeline_config configs/ltxv-13b-0.9.8-distilled.yaml
```

## Why Use This?

- **Save VRAM**: Skip loading the text encoder (~3GB) during video generation
- **Reuse Embeddings**: Generate multiple videos with the same prompt efficiently
- **Separate Encoding**: Run text encoding on a different machine/GPU than video generation
