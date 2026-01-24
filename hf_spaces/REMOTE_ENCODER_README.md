# Remote Text Encoder Offloading

This feature allows the LTX Video spaces to offload text encoding to a dedicated encoder space, reducing memory usage and allowing for more efficient deployment.

## Overview

The remote text encoder architecture consists of three main components:

1. **LTX-2 Text Encoder Space** (`hf_spaces/ltx-2-text-encoder/`): 
   - Dedicated space that hosts the large Gemma-3 12B text encoder
   - Provides both UI and API endpoints for encoding text prompts
   - Supports both positive and negative prompts
   - Returns video and audio context embeddings

2. **LTX Video Distilled Tester Space** (`hf_spaces/ltx-video-distilled-tester/`):
   - Consumer space that can use either local or remote text encoding
   - When using remote encoding, skips loading the local T5 text encoder
   - Saves significant GPU memory by offloading text encoding

3. **SD3.5 Large LoRA Space** (`hf_spaces/sd3.5-large-lora/`):
   - Note: Limited support for remote encoding due to integrated text encoders in diffusers pipeline
   - For better memory optimization, consider using the LTX Video Distilled Tester space

## Setup Instructions

### 1. Deploy the LTX-2 Text Encoder Space

First, deploy the encoder space on Hugging Face Spaces:

```bash
cd hf_spaces/ltx-2-text-encoder
# Push to your Hugging Face Space
# e.g., https://huggingface.co/spaces/YOUR_USERNAME/ltx-2-text-encoder
```

Note the URL of your deployed encoder space (e.g., `YOUR_USERNAME/ltx-2-text-encoder`).

### 2. Configure Consumer Spaces

#### For LTX Video Distilled Tester:

Set the following environment variables in your Hugging Face Space settings:

```bash
USE_REMOTE_TEXT_ENCODER=true
REMOTE_ENCODER_SPACE_URL=YOUR_USERNAME/ltx-2-text-encoder
```

#### For SD3.5 Large LoRA:

Set the following environment variable (limited support):

```bash
USE_REMOTE_TEXT_ENCODER=true
```

## How It Works

### Text Encoder Space

The encoder space loads the Gemma-3 12B model and provides two endpoints:

1. **File-based encoding** (`/encode`): Encodes prompts and saves embeddings to `.pt` files
2. **API endpoint** (`/encode_api`): Returns embeddings directly for remote clients

### Consumer Spaces

When `USE_REMOTE_TEXT_ENCODER=true`:

1. The consumer space **skips loading** its local text encoder model
2. During inference, it calls the remote encoder space via Gradio Client API
3. Receives pre-computed embeddings from the encoder space
4. Uses these embeddings for video/image generation

### Fallback Behavior

The `remote_text_encoder.py` utility provides automatic fallback:

- First attempts to use remote encoding
- If remote encoding fails, falls back to local encoding (if available)
- Logs all encoding operations for debugging

## Memory Savings

Using remote text encoding provides significant memory savings:

| Component | Without Remote | With Remote | Savings |
|-----------|---------------|-------------|---------|
| LTX Distilled Tester | ~24GB VRAM | ~16GB VRAM | ~8GB |
| T5 Encoder | Loaded locally | Not loaded | 100% |

## Architecture Diagram

```
┌─────────────────────────────────────┐
│  LTX-2 Text Encoder Space           │
│  (Gemma-3 12B)                      │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  encode_prompt_api()         │  │
│  │  - Accepts: prompt, neg_prompt│ │
│  │  - Returns: embeddings dict   │  │
│  └──────────────────────────────┘  │
└──────────────┬──────────────────────┘
               │ Gradio Client API
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌─────────────┐ ┌─────────────────────┐
│ LTX Video   │ │ SD3.5 Large LoRA    │
│ Distilled   │ │ (limited support)   │
│ Tester      │ │                     │
│             │ │                     │
│ - No local  │ │ - Integrated        │
│   T5 encoder│ │   encoders          │
│ - Uses      │ │                     │
│   remote    │ │                     │
└─────────────┘ └─────────────────────┘
```

## API Usage Example

### Direct API Call to Encoder Space

```python
from gradio_client import Client

# Connect to encoder space
client = Client("YOUR_USERNAME/ltx-2-text-encoder")

# Encode a prompt
result = client.predict(
    prompt="An astronaut floating in space",
    negative_prompt="blurry, low quality",
    api_name="/encode_api"
)

embeddings, status = result
print(f"Status: {status}")
print(f"Video context shape: {embeddings['video_context'].shape}")
```

### Using the Remote Encoder Client

```python
from remote_text_encoder import RemoteTextEncoderClient

# Initialize client
client = RemoteTextEncoderClient("YOUR_USERNAME/ltx-2-text-encoder")

# Encode prompt
video_ctx, audio_ctx, video_neg, audio_neg = client.encode_prompt(
    prompt="An astronaut floating in space",
    negative_prompt="blurry, low quality",
    device="cuda"
)

# Use embeddings in your pipeline
# ...
```

## Testing

### Test the Encoder Space

1. Deploy the encoder space
2. Use the "API Endpoint" tab to test encoding
3. Verify that embeddings are returned with correct shapes

### Test Remote Encoding

1. Set environment variables in consumer space
2. Check logs for "⚡ REMOTE TEXT ENCODER MODE ENABLED"
3. Verify that text encoder loading is skipped
4. Monitor memory usage during inference

## Troubleshooting

### Connection Issues

If the consumer space cannot connect to the encoder:

```
❌ Failed to connect to remote encoder: ...
```

**Solutions:**
- Verify the encoder space URL is correct
- Check that the encoder space is running
- Ensure both spaces are public or have proper access

### Embedding Format Issues

If embeddings don't have the expected format:

```
Warning: Failed to deserialize tensor: ...
```

**Solutions:**
- Update both encoder and consumer spaces to latest version
- Check Gradio Client compatibility
- Verify tensor serialization format

### Fallback to Local Encoding

If you see:

```
⚠️ Remote encoding failed, trying fallback: ...
🔄 Using local text encoder
```

The system is working correctly by falling back to local encoding.

## Performance Considerations

### Latency

Remote encoding adds network latency:
- Local encoding: ~0.5-2s
- Remote encoding: ~1-3s (includes network overhead)

### Throughput

For batch processing:
- Consider caching embeddings for repeated prompts
- Use the file-based encoding endpoint for offline workflows
- Remote encoding is best for interactive use cases

### Cost

Remote encoding allows:
- Smaller GPU instances for consumer spaces
- One powerful instance for the encoder space
- Better resource utilization across multiple consumers

## Future Enhancements

Potential improvements:

1. **Caching**: Add embedding cache to reduce repeated encoding
2. **Batch API**: Support batch encoding for multiple prompts
3. **T5 Support**: Add T5 encoder support alongside Gemma-3
4. **Compression**: Compress embeddings for faster transfer
5. **Load Balancing**: Support multiple encoder instances

## License

See the main repository LICENSE file.
