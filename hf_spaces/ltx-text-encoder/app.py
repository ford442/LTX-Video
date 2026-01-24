"""
LTX-Video Text Encoder Space
Encodes text prompts using T5 for LTX-Video generation.
This allows pre-computing embeddings to skip text encoder loading in the main pipeline.
"""
import time
from pathlib import Path
import torch
import gradio as gr
from transformers import T5EncoderModel, T5Tokenizer

# HuggingFace Hub defaults for LTX-Video text encoder
DEFAULT_TEXT_ENCODER_REPO = "PixArt-alpha/PixArt-XL-2-1024-MS"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Initializing T5 text encoder with:")
print(f"  model={DEFAULT_TEXT_ENCODER_REPO}")
print(f"  device={device}")

# Load text encoder and tokenizer
text_encoder = T5EncoderModel.from_pretrained(
    DEFAULT_TEXT_ENCODER_REPO, subfolder="text_encoder"
)
tokenizer = T5Tokenizer.from_pretrained(
    DEFAULT_TEXT_ENCODER_REPO, subfolder="tokenizer"
)

text_encoder = text_encoder.to(device)
text_encoder = text_encoder.to(torch.bfloat16)
text_encoder.eval()

print("=" * 80)
print("T5 Text encoder loaded and ready!")
print("=" * 80)


@torch.inference_mode()
def encode_prompt(
    prompt: str,
    negative_prompt: str = "",
    max_length: int = 256,
):
    """
    Encode a text prompt using T5 text encoder for LTX-Video.
    
    Args:
        prompt: The positive text prompt
        negative_prompt: The negative text prompt (optional)
        max_length: Maximum token length for encoding
    
    Returns:
        Path to saved embeddings file and status message
    """
    start_time = time.time()

    try:
        # Encode the positive prompt
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)
        
        # Get embeddings
        prompt_embeds = text_encoder(
            text_input_ids, attention_mask=prompt_attention_mask
        )[0]

        # Encode negative prompt if provided
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
        if negative_prompt:
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_prompt_attention_mask = uncond_input.attention_mask.to(device)
            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=negative_prompt_attention_mask,
            )[0]

        # Output directory setup
        output_dir = Path("embeddings")
        output_dir.mkdir(exist_ok=True)
        
        # Create a clean filename from the prompt (first 30 chars, safe chars only)
        safe_name = "".join([c for c in prompt[:30] if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_')
        output_path = output_dir / f"ltx_emb_{safe_name}_{int(time.time())}.pt"

        # Prepare data dict
        embedding_data = {
            'prompt_embeds': prompt_embeds.cpu(),
            'prompt_attention_mask': prompt_attention_mask.cpu(),
            'prompt': prompt,
        }

        # Add negative contexts if they were encoded
        if negative_prompt_embeds is not None:
            embedding_data['negative_prompt_embeds'] = negative_prompt_embeds.cpu()
            embedding_data['negative_prompt_attention_mask'] = negative_prompt_attention_mask.cpu()
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
with gr.Blocks(title="LTX-Video T5 Text Encoder") as demo:
    gr.Markdown("# LTX-Video T5 Text Encoder 🎯")
    gr.Markdown("""
    **Standalone Encoder:** Encodes prompts into T5 embeddings for LTX-Video. 
    Use these pre-computed embeddings to skip text encoder loading in the main pipeline.
    
    **Usage:**
    1. Enter your prompt and optionally a negative prompt
    2. Click "Encode Prompt" to generate embeddings
    3. Download the .pt file
    4. Use it with LTX-Video inference by passing `--embeddings_path /path/to/embeddings.pt`
    """)

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=5,
                value="A serene lake surrounded by mountains at sunset, with reflections on the water"
            )

            negative_prompt_input = gr.Textbox(
                label="Negative Prompt (Optional)",
                placeholder="Enter negative prompt...",
                lines=2,
                value="worst quality, inconsistent motion, blurry, jittery, distorted"
            )
            
            max_length_input = gr.Slider(
                label="Max Token Length",
                minimum=64,
                maximum=512,
                value=256,
                step=64,
                info="Maximum number of tokens for encoding"
            )

            encode_btn = gr.Button("Encode Prompt", variant="primary", size="lg")

        with gr.Column():
            embedding_file = gr.File(label="Embedding File (.pt)")
            status_output = gr.Textbox(label="Status", lines=2)

    encode_btn.click(
        fn=encode_prompt,
        inputs=[prompt_input, negative_prompt_input, max_length_input],
        outputs=[embedding_file, status_output]
    )
    
    gr.Markdown("""
    ### Example Commands
    
    After downloading the embeddings file, you can use it with LTX-Video:
    
    ```bash
    # Text-to-video with pre-computed embeddings
    python inference.py \\
        --embeddings_path ltx_emb_yourprompt.pt \\
        --height 704 --width 1216 --num_frames 121 \\
        --seed 42 --pipeline_config configs/ltxv-13b-0.9.8-distilled.yaml
    
    # Image-to-video with pre-computed embeddings
    python inference.py \\
        --embeddings_path ltx_emb_yourprompt.pt \\
        --conditioning_media_paths image.jpg \\
        --conditioning_start_frames 0 \\
        --height 704 --width 1216 --num_frames 121 \\
        --seed 42 --pipeline_config configs/ltxv-13b-0.9.8-distilled.yaml
    ```
    """)

css = '''
.gradio-container .contain{max-width: 1200px !important; margin: 0 auto !important}
'''

if __name__ == "__main__":
    demo.launch(css=css)
