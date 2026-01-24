"""
LTX-2 Gemma Text Encoder Space (Streamlined)
Encodes text prompts using Gemma-3-12B for LTX-2 video generation.
Removes prompt enhancement for pure encoding speed.
"""
import time
from pathlib import Path
import numpy as np
import spaces
import gradio as gr
import torch
from huggingface_hub import hf_hub_download, snapshot_download

# Import from public LTX-2 package
from ltx_pipelines.utils import ModelLedger

# HuggingFace Hub defaults
DEFAULT_REPO_ID = "Lightricks/LTX-2"
DEFAULT_GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"
DEFAULT_CHECKPOINT_FILENAME = "ltx-2-19b-dev-fp8.safetensors"

def get_hub_or_local_checkpoint(repo_id: str, filename: str):
    """Download from HuggingFace Hub."""
    print(f"Downloading {filename} from {repo_id}...")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Downloaded to {ckpt_path}")
    return ckpt_path

def download_gemma_model(repo_id: str):
    """Download the full Gemma model directory."""
    print(f"Downloading Gemma model from {repo_id}...")
    local_dir = snapshot_download(repo_id=repo_id)
    print(f"Gemma model downloaded to {local_dir}")
    return local_dir


checkpoint_path = get_hub_or_local_checkpoint(DEFAULT_REPO_ID, DEFAULT_CHECKPOINT_FILENAME)
#gemma_local_path = download_gemma_model(DEFAULT_GEMMA_REPO_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Initializing text encoder with:")
print(f"  checkpoint_path={checkpoint_path}")
#print(f"  gemma_root={gemma_local_path}")
print(f"  device={device}")

# We use bfloat16 by default as it is the native training dtype for Gemma 3
model_ledger = ModelLedger(
    dtype=torch.bfloat16,
    device=device,
    checkpoint_path=checkpoint_path,
    gemma_root_path=DEFAULT_GEMMA_REPO_ID,
    local_files_only=False
)

# Load text encoder once and keep it in memory
text_encoder = model_ledger.text_encoder()

print("=" * 80)
print("Text encoder loaded and ready!")
print("=" * 80)

def encode_text_simple(text_encoder, prompt: str):
    """Simple text encoding without using pipeline_utils."""
    # Gemma 3 encoder returns video_context, audio_context, and pooled_embedding
    # We only need the contexts for LTX-2
    v_context, a_context, _ = text_encoder(prompt)
    return v_context, a_context

@spaces.GPU()
@torch.inference_mode()  # Optimizes memory and speed by disabling gradient tracking
def encode_prompt(
    prompt: str,
    negative_prompt: str = ""
):
    """
    Encode a text prompt using Gemma text encoder.
    """
    start_time = time.time()

    try:
        # Encode the positive prompt
        video_context, audio_context = encode_text_simple(text_encoder, prompt)

        # Encode negative prompt if provided
        video_context_negative = None
        audio_context_negative = None
        if negative_prompt:
            video_context_negative, audio_context_negative = encode_text_simple(text_encoder, negative_prompt)

        # Output directory setup
        output_dir = Path("embeddings")
        output_dir.mkdir(exist_ok=True)
        
        # Create a clean filename from the prompt (first 30 chars, safe chars only)
        safe_name = "".join([c for c in prompt[:30] if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_')
        output_path = output_dir / f"emb_{safe_name}_{int(time.time())}.pt"

        # Prepare data dict
        embedding_data = {
            'video_context': video_context.cpu(),
            'audio_context': audio_context.cpu(),
            'prompt': prompt,
            'original_prompt': prompt, # Kept for compatibility
        }

        # Add negative contexts if they were encoded
        if video_context_negative is not None:
            embedding_data['video_context_negative'] = video_context_negative.cpu()
            embedding_data['audio_context_negative'] = audio_context_negative.cpu()
            embedding_data['negative_prompt'] = negative_prompt

        torch.save(embedding_data, output_path)

        # Get memory stats
        elapsed_time = time.time() - start_time
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            status = f"✓ Encoded in {elapsed_time:.2f}s | VRAM Alloc: {allocated:.2f}GB"
        else:
            status = f"✓ Encoded in {elapsed_time:.2f}s (CPU mode)"

        return str(output_path), status

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

# Create Gradio interface
with gr.Blocks(title="LTX-2 Gemma Text Encoder (Pure)") as demo:
    gr.Markdown("# LTX-2 Gemma Text Encoder (Pure) 🎯")
    gr.Markdown("""
    **Standalone Encoder:** Encodes prompts into embeddings for LTX-2. 
    Does not perform prompt enhancement/expansion.
    """)

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=5,
                value="An astronaut hatches from a fragile egg on the surface of the Moon"
            )

            negative_prompt_input = gr.Textbox(
                label="Negative Prompt (Optional)",
                placeholder="Enter negative prompt...",
                lines=2,
                value=""
            )

            encode_btn = gr.Button("Encode Prompt", variant="primary", size="lg")

        with gr.Column():
            embedding_file = gr.File(label="Embedding File (.pt)")
            status_output = gr.Textbox(label="Status", lines=1)

    encode_btn.click(
        fn=encode_prompt,
        inputs=[prompt_input, negative_prompt_input],
        outputs=[embedding_file, status_output]
    )

css = '''
.gradio-container .contain{max-width: 1200px !important; margin: 0 auto !important}
'''

if __name__ == "__main__":
    demo.launch(css=css)