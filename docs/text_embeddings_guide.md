# Pre-computed Text Embeddings for LTX-Video

This document explains how to use pre-computed text embeddings with LTX-Video to optimize VRAM usage and enable flexible workflows.

## Overview

The standard LTX-Video workflow loads a T5 text encoder (~3GB VRAM) alongside the transformer and VAE models. By pre-computing text embeddings, you can:

- **Save VRAM**: Skip loading the text encoder during video generation
- **Reuse embeddings**: Encode once, generate multiple videos with the same prompt
- **Separate workflows**: Run text encoding on one machine/GPU and video generation on another
- **Batch processing**: Pre-encode many prompts and generate videos later

## Quick Start

### 1. Encode Text Prompts

Use the standalone text encoder space or script:

```python
from transformers import T5EncoderModel, T5Tokenizer
import torch

# Load T5 encoder
text_encoder = T5EncoderModel.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS", 
    subfolder="text_encoder"
)
tokenizer = T5Tokenizer.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    subfolder="tokenizer"
)

# Encode prompt
prompt = "A serene lake surrounded by mountains"
text_inputs = tokenizer(
    prompt, padding="max_length", max_length=256, 
    truncation=True, return_tensors="pt"
)
prompt_embeds = text_encoder(text_inputs.input_ids)[0]

# Save embeddings
torch.save({
    'prompt_embeds': prompt_embeds.cpu(),
    'prompt_attention_mask': text_inputs.attention_mask.cpu(),
    'prompt': prompt
}, 'my_embeddings.pt')
```

### 2. Generate Video with Pre-computed Embeddings

```bash
python inference.py \
    --embeddings_path my_embeddings.pt \
    --height 704 --width 1216 --num_frames 121 \
    --seed 42 \
    --pipeline_config configs/ltxv-13b-0.9.8-distilled.yaml
```

## Text Encoder Space

A Gradio app is provided in `hf_spaces/ltx-text-encoder/` for easy text encoding:

### Running the Text Encoder Space

```bash
cd hf_spaces/ltx-text-encoder
pip install -r requirements.txt
python app.py
```

This launches a web interface where you can:
1. Enter your prompt and negative prompt
2. Click "Encode Prompt"
3. Download the generated `.pt` file
4. Use it with LTX-Video inference

## Video Stitching with Embeddings

The included Colab notebook demonstrates how to:
1. Encode multiple prompts
2. Generate video clips with each prompt
3. Stitch clips together into a longer video

### Basic Stitching

```python
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Load clips
clips = [VideoFileClip(path) for path in video_paths]

# Concatenate
final = concatenate_videoclips(clips, method="compose")
final.write_videofile("stitched.mp4")
```

### Advanced: Image-to-Video Stitching

For smoother transitions, use the last frame of each clip as the conditioning image for the next:

```python
import cv2

def extract_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, frame = cap.read()
    cap.release()
    return frame

# Generate clip 1
infer(embeddings_path="clip1.pt", ...)

# Extract last frame
last_frame = extract_last_frame("output1.mp4")
cv2.imwrite("frame.jpg", last_frame)

# Generate clip 2 conditioned on clip 1's last frame
infer(
    embeddings_path="clip2.pt",
    conditioning_media_paths=["frame.jpg"],
    conditioning_start_frames=[0],
    ...
)
```

## API Reference

### `load_embeddings_from_file(embeddings_path, device=None)`

Load pre-computed text embeddings from a `.pt` file.

**Parameters:**
- `embeddings_path` (str): Path to the embeddings file
- `device` (str, optional): Device to load embeddings to

**Returns:**
- dict containing:
  - `prompt_embeds`: Tensor of shape (1, seq_len, hidden_dim)
  - `negative_prompt_embeds`: Tensor (if negative prompt was encoded)
  - `prompt_attention_mask`: Attention mask for prompt
  - `negative_prompt_attention_mask`: Attention mask for negative prompt
  - `prompt`: Original text prompt
  - `negative_prompt`: Original negative prompt

### `create_ltx_video_pipeline(..., skip_text_encoder=False)`

Create an LTX-Video pipeline with optional text encoder skipping.

**Parameters:**
- `skip_text_encoder` (bool): If True, text encoder won't be loaded. Pre-computed embeddings must be provided during inference.

### Command-Line Arguments

```
--embeddings_path PATH
    Path to pre-computed text embeddings file (.pt). 
    If provided, text encoder loading will be skipped.
```

## Embedding File Format

The `.pt` file should contain a dictionary with the following keys:

```python
{
    'prompt_embeds': torch.Tensor,              # Required
    'prompt_attention_mask': torch.Tensor,      # Required
    'prompt': str,                               # Optional but recommended
    'negative_prompt_embeds': torch.Tensor,     # Optional
    'negative_prompt_attention_mask': torch.Tensor,  # Optional
    'negative_prompt': str,                      # Optional
}
```

## Performance Comparison

### Standard Workflow
- Load transformer: ~15GB VRAM
- Load VAE: ~2GB VRAM
- Load text encoder: ~3GB VRAM
- **Total: ~20GB VRAM**

### With Pre-computed Embeddings
- Load transformer: ~15GB VRAM
- Load VAE: ~2GB VRAM
- Load embeddings: ~0.01GB VRAM
- **Total: ~17GB VRAM**

**Savings: 3GB VRAM (~15%)**

## Colab Notebook

A complete Colab notebook is included: `ltx_video_embeddings_stitching_colab.ipynb`

The notebook demonstrates:
- Text encoding with T5
- Video generation with pre-computed embeddings
- Multi-clip generation
- Video stitching
- Advanced image-to-video conditioning for smooth transitions

## Troubleshooting

### Error: "You should provide either prompt_embeds or self.text_encoder should not be None"

This means you're trying to use pre-computed embeddings but they're not being loaded correctly. Check:
1. The embeddings file exists and is readable
2. The file contains the required keys (`prompt_embeds`, `prompt_attention_mask`)
3. The embeddings are on the correct device

### Error: Shape mismatch

Ensure the embeddings were encoded with the same max_length as expected by the pipeline (default: 256 tokens).

### Poor quality results

Pre-computed embeddings should produce identical results to on-the-fly encoding. If quality differs:
1. Verify you're using the same text encoder model
2. Check that embeddings are in the correct dtype (bfloat16)
3. Ensure attention masks are included

## Examples

See `ltx_video_embeddings_stitching_colab.ipynb` for complete examples including:
- Basic text-to-video with embeddings
- Image-to-video with embeddings
- Multi-clip generation and stitching
- Advanced workflows with frame-by-frame conditioning
