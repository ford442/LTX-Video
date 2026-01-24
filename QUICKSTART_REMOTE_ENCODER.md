# Quick Start Guide: Remote Text Encoder Offloading

This guide helps you quickly set up and use the remote text encoder feature.

## What This Does

Allows LTX video generation spaces to **offload text encoding** to a dedicated encoder space, saving ~8GB of VRAM on consumer spaces.

```
┌────────────────┐
│ Encoder Space  │  ← Hosts Gemma-3 12B (13-15GB VRAM)
│ (1 instance)   │
└────────┬───────┘
         │ API
    ┌────┴────┐
    │         │
┌───▼────┐ ┌──▼──────┐
│ Space1 │ │ Space2  │  ← Consumers (saves 8GB each)
└────────┘ └─────────┘
```

## Quick Setup (3 Steps)

### Step 1: Deploy the Encoder Space

1. Go to Hugging Face Spaces: https://huggingface.co/spaces
2. Create a new Space called `ltx-2-text-encoder`
3. Copy contents from `hf_spaces/ltx-2-text-encoder/` to your Space
4. Deploy and wait for it to load the model
5. Note your Space URL (e.g., `YOUR_USERNAME/ltx-2-text-encoder`)

### Step 2: Configure Consumer Space

For **LTX Video Distilled Tester**:

1. Go to your Space settings → Variables
2. Add these environment variables:
   ```
   USE_REMOTE_TEXT_ENCODER = true
   REMOTE_ENCODER_SPACE_URL = YOUR_USERNAME/ltx-2-text-encoder
   ```
3. Restart the Space

### Step 3: Verify It Works

1. Check Space logs for: `⚡ REMOTE TEXT ENCODER MODE ENABLED`
2. Should also see: `⚡ Skipping local text encoder loading`
3. Memory usage should be ~8GB lower
4. Generate a video to test end-to-end

## What Gets Modified

### Files Changed:

| File | Changes |
|------|---------|
| `hf_spaces/ltx-2-text-encoder/app.py` | Added API endpoint for remote calls |
| `hf_spaces/ltx-video-distilled-tester/inference.py` | Added `use_remote_text_encoder` parameter |
| `hf_spaces/ltx-video-distilled-tester/app.py` | Check env vars and skip local encoder |
| `hf_spaces/ltx-video-distilled-tester/remote_text_encoder.py` | Client for calling remote encoder |
| `hf_spaces/sd3.5-large-lora/app.py` | Limited support (see note below) |

### New Files:

- `hf_spaces/remote_text_encoder.py` - Shared client utility
- `hf_spaces/REMOTE_ENCODER_README.md` - Full documentation
- `hf_spaces/.env.example` - Configuration examples
- `test_remote_encoder.py` - Test suite

## Memory Savings

| Configuration | VRAM Usage | Notes |
|--------------|------------|-------|
| **Before** (local encoding) | ~24GB | T5 encoder + models |
| **After** (remote encoding) | ~16GB | Models only, encoder offloaded |
| **Encoder Space** | ~13-15GB | Gemma-3 12B |

**Total savings**: ~8GB per consumer space (multiple consumers can share one encoder)

## Performance Impact

- **Latency**: Adds ~0.5-1s network overhead per generation
- **Throughput**: Suitable for interactive use, not batch processing
- **Reliability**: Automatic fallback to local encoding if remote fails

## Important Notes

### ✅ Full Support
- **LTX Video Distilled Tester** - Recommended, works perfectly

### ⚠️ Limited Support  
- **SD3.5 Large LoRA** - Text encoders are integrated into diffusers pipeline, hard to offload

### 🔧 For SD3.5 LoRA Users
The SD3.5 pipeline has multiple integrated text encoders (CLIP + T5) that are tightly coupled. While we added environment variable support, actual offloading is limited. Consider using LTX Video Distilled Tester for better memory optimization.

## Troubleshooting

### "Failed to connect to remote encoder"

**Cause**: Cannot reach the encoder space

**Fix**:
```bash
# Check encoder space is running
# Verify URL is correct (no typos)
# Ensure spaces are public or have access
# Test manually: https://huggingface.co/spaces/YOUR_USERNAME/ltx-2-text-encoder
```

### "Skipping remote encoder - not available"

**Cause**: Environment variables not set

**Fix**:
```bash
# Add to Space settings → Variables:
USE_REMOTE_TEXT_ENCODER=true
REMOTE_ENCODER_SPACE_URL=YOUR_USERNAME/ltx-2-text-encoder

# Restart the Space
```

### Encoding is slow

**Cause**: Network latency or encoder is cold-starting

**Fix**:
- Encoder space may be sleeping - call it once to wake up
- Check network connection
- Consider deploying encoder in same region
- For batch work, use local encoding instead

### "Failed to deserialize tensor"

**Cause**: Version mismatch or network error

**Fix**:
- Update both spaces to latest version
- Check gradio_client is installed
- Verify network is stable during encoding
- Try again (may be transient)

## Testing

Run the test suite to verify everything works:

```bash
# Test API function (requires encoder space running)
python test_remote_encoder.py

# Test with specific encoder URL
REMOTE_ENCODER_SPACE_URL=YOUR_USERNAME/ltx-2-text-encoder python test_remote_encoder.py
```

Expected output:
```
✅ API function test passed!
✅ Remote client test passed!
✅ All tests passed!
```

## API Usage Examples

### Direct Gradio Client Call

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/ltx-2-text-encoder")

result = client.predict(
    prompt="An astronaut floating in space",
    negative_prompt="blurry, low quality",
    api_name="/encode_api"
)

embeddings, status = result
print(f"Video context shape: {embeddings['video_context_shape']}")
```

### Using the Client Utility

```python
from remote_text_encoder import RemoteTextEncoderClient

client = RemoteTextEncoderClient("YOUR_USERNAME/ltx-2-text-encoder")

video_ctx, audio_ctx, video_neg, audio_neg = client.encode_prompt(
    prompt="An astronaut floating in space",
    negative_prompt="blurry, low quality",
    device="cuda"
)

# Use embeddings in your pipeline...
```

## Cost Optimization

### Single Consumer
- **Without offload**: 1x A100 (24GB) = ~$1/hour
- **With offload**: 1x A100 (16GB) = ~$0.70/hour + encoder

Not cost-effective for single consumer.

### Multiple Consumers
- **3 consumers without offload**: 3x A100 (24GB) = ~$3/hour
- **3 consumers with offload**: 3x A100 (16GB) + 1x A100 (15GB) = ~$2.60/hour

**Savings**: ~13% with 3+ consumers sharing one encoder

## Next Steps

1. ✅ Deploy encoder space
2. ✅ Configure consumer space(s)
3. ✅ Test with one video generation
4. ✅ Monitor memory usage
5. ✅ Scale to multiple consumers if needed

## Support

- Full docs: `hf_spaces/REMOTE_ENCODER_README.md`
- Configuration: `hf_spaces/.env.example`
- Tests: `test_remote_encoder.py`

## License

See main repository LICENSE.
