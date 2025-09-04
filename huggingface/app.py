from moviepy.editor import VideoFileClip, concatenate_videoclips
import spaces
import os
os.putenv('PYTORCH_NVML_BASED_CUDA_CHECK','1')
os.putenv('TORCH_LINALG_PREFER_CUSOLVER','1')
alloc_conf_parts = [
    'expandable_segments:True',
    'pinned_use_background_threads:True'
]
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ','.join(alloc_conf_parts)
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.putenv('HF_HUB_ENABLE_HF_TRANSFER','1')

import torch
import cv2
import gc
import copy
import inspect
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.preferred_blas_library="cublas"
torch.backends.cuda.preferred_linalg_library="cusolver"
torch.set_float32_matmul_precision("highest")

import gradio as gr
import numpy as np
import random
import yaml
from pathlib import Path
import imageio
import tempfile
from PIL import Image
from huggingface_hub import hf_hub_download
import shutil

MAX_SEED = np.iinfo(np.int32).max

from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

print("Loading SDXL Image-to-Image pipeline...")
enhancer_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "ford442/stable-diffusion-xl-refiner-1.0-bf16",
    requires_aesthetics_score=True,
)
enhancer_pipeline.vae.set_default_attn_processor()
enhancer_pipeline.to("cpu")
print("SDXL Image-to-Image pipeline loaded successfully.")

from inference import (
    create_ltx_video_pipeline,
    create_latent_upsampler,
    load_image_to_tensor_with_resize_and_crop,
    seed_everething,
    get_device,
    calculate_padding,
    load_media_file,
)
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline, ConditioningItem
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.models.autoencoders.vae_encode import (
    un_normalize_latents,
    normalize_latents,
)


def adain_filter_latent(
    latents: torch.Tensor, reference_latents: torch.Tensor, factor=1.0
):
    """
    Applies Adaptive Instance Normalization (AdaIN) to a latent tensor based on
    statistics from a reference latent tensor.

    Args:
        latent (torch.Tensor): Input latents to normalize
        reference_latent (torch.Tensor): The reference latents providing style statistics.
        factor (float): Blending factor between original and transformed latent.
                       Range: -10.0 to 10.0, Default: 1.0

    Returns:
        torch.Tensor: The transformed latent tensor
    """
    result = latents.clone()

    for i in range(latents.size(0)):
        for c in range(latents.size(1)):
            r_sd, r_mean = torch.std_mean(
                reference_latents[i, c], dim=None
            )  # index by original dim order
            i_sd, i_mean = torch.std_mean(result[i, c], dim=None)

            result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean

    result = torch.lerp(latents, result, factor)
    return result


class CorrectedLTXMultiScalePipeline:
    def _upsample_latents(
        self, latest_upsampler: LatentUpsampler, latents: torch.Tensor
    ):
        assert latents.device == latest_upsampler.device

        latents = un_normalize_latents(
            latents, self.vae, vae_per_channel_normalize=True
        )
        upsampled_latents = latest_upsampler(latents)
        upsampled_latents = normalize_latents(
            upsampled_latents, self.vae, vae_per_channel_normalize=True
        )
        return upsampled_latents

    def __init__(
        self, video_pipeline: LTXVideoPipeline, latent_upsampler: LatentUpsampler
    ):
        self.video_pipeline = video_pipeline
        self.vae = video_pipeline.vae
        self.latent_upsampler = latent_upsampler

    def __call__(
        self,
        downscale_factor: float,
        first_pass: dict,
        second_pass: dict,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        original_kwargs = kwargs.copy()
        original_output_type = kwargs["output_type"]
        original_width = kwargs["width"]
        original_height = kwargs["height"]

        x_width = int(kwargs["width"] * downscale_factor)
        downscaled_width = x_width - (x_width % self.video_pipeline.vae_scale_factor)
        x_height = int(kwargs["height"] * downscale_factor)
        downscaled_height = x_height - (x_height % self.video_pipeline.vae_scale_factor)

        kwargs["output_type"] = "latent"
        kwargs["width"] = downscaled_width
        kwargs["height"] = downscaled_height
        kwargs.update(**first_pass)
        result = self.video_pipeline(*args, **kwargs)
        latents = result.images

        upsampled_latents = self._upsample_latents(self.latent_upsampler, latents)
        upsampled_latents = adain_filter_latent(
            latents=upsampled_latents, reference_latents=latents
        )

        kwargs = original_kwargs

        kwargs["latents"] = upsampled_latents
        kwargs["output_type"] = original_output_type
        kwargs["width"] = downscaled_width * 2
        kwargs["height"] = downscaled_height * 2
        kwargs.update(**second_pass)

        result = self.video_pipeline(*args, **kwargs)
        if original_output_type != "latent":
            num_frames = result.images.shape[2]
            videos = rearrange(result.images, "b c f h w -> (b f) c h w")

            videos = F.interpolate(
                videos,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            videos = rearrange(videos, "(b f) c h w -> b c f h w", f=num_frames)
            result.images = videos

        return result
# All other custom classes and functions will be defined at the end of the file.

config_file_path = "configs/ltxv-13b-0.9.8-distilled.yaml"
with open(config_file_path, "r") as file:
    PIPELINE_CONFIG_YAML = yaml.safe_load(file)

LTX_REPO = "Lightricks/LTX-Video"
MAX_IMAGE_SIZE = PIPELINE_CONFIG_YAML.get("max_resolution", 1280)
MAX_NUM_FRAMES = 900
N_LATENT_OVERLAP_FRAMES = 2

pipeline_instance = None
latent_upsampler_instance = None
models_dir = "downloaded_models_gradio_cpu_init"
Path(models_dir).mkdir(parents=True, exist_ok=True)
print("Downloading models (if not present)...")
distilled_model_actual_path = hf_hub_download(repo_id=LTX_REPO, filename=PIPELINE_CONFIG_YAML["checkpoint_path"], local_dir=models_dir, local_dir_use_symlinks=False)
PIPELINE_CONFIG_YAML["checkpoint_path"] = distilled_model_actual_path
SPATIAL_UPSCALER_FILENAME = PIPELINE_CONFIG_YAML["spatial_upscaler_model_path"]
spatial_upscaler_actual_path = hf_hub_download(repo_id=LTX_REPO, filename=SPATIAL_UPSCALER_FILENAME, local_dir=models_dir, local_dir_use_symlinks=False)
PIPELINE_CONFIG_YAML["spatial_upscaler_model_path"] = spatial_upscaler_actual_path
print("Creating LTX Video pipeline on CPU...")
pipeline_instance = create_ltx_video_pipeline(ckpt_path=PIPELINE_CONFIG_YAML["checkpoint_path"], precision=PIPELINE_CONFIG_YAML["precision"], text_encoder_model_name_or_path=PIPELINE_CONFIG_YAML["text_encoder_model_name_or_path"], sampler=PIPELINE_CONFIG_YAML["sampler"], device="cpu", enhance_prompt=False, prompt_enhancer_image_caption_model_name_or_path=PIPELINE_CONFIG_YAML["prompt_enhancer_image_caption_model_name_or_path"], prompt_enhancer_llm_model_name_or_path=PIPELINE_CONFIG_YAML["prompt_enhancer_llm_model_name_or_path"])
if PIPELINE_CONFIG_YAML.get("spatial_upscaler_model_path"):
    print("Creating latent upsampler on CPU...")
    latent_upsampler_instance = create_latent_upsampler(PIPELINE_CONFIG_YAML["spatial_upscaler_model_path"], device="cpu")
target_inference_device = "cuda"
print(f"Target inference device: {target_inference_device}")
pipeline_instance.to(target_inference_device)
if latent_upsampler_instance: latent_upsampler_instance.to(target_inference_device)


def calculate_new_dimensions(orig_w, orig_h):
    if orig_w == 0 or orig_h == 0: return int(768), int(768)
    if orig_w >= orig_h:
        new_h, new_w = 768, round((768 * (orig_w / orig_h)) / 32) * 32
    else:
        new_w, new_h = 768, round((768 * (orig_h / orig_w)) / 32) * 32
    return int(max(256, min(new_h, MAX_IMAGE_SIZE))), int(max(256, min(new_w, MAX_IMAGE_SIZE)))


def get_duration(*args, **kwargs):
    duration_ui = kwargs.get('duration_ui', 5.0)
    if duration_ui > 7.0: return 120
    if duration_ui > 5.0: return 100
    if duration_ui > 3.0: return 90
    if duration_ui > 2.0: return 90
    if duration_ui > 1.5: return 60
    if duration_ui > 1.0: return 45
    if duration_ui > 0.5: return 30
    return 20


@spaces.GPU(duration=30)
def enhance_frame(image_to_enhance: Image.Image, refine_prompt: str, refine_strength: float, refine_steps: int):
    try:
        print("Moving enhancer pipeline to GPU...")
        seed = random.randint(0, MAX_SEED)
        generator = torch.Generator(device='cuda').manual_seed(seed)
        enhancer_pipeline.to("cuda",torch.bfloat16)
        print(f"Refining frame with prompt: '{refine_prompt}', strength: {refine_strength}, steps: {refine_steps}")
        enhanced_image = enhancer_pipeline(prompt=refine_prompt, image=image_to_enhance, strength=refine_strength, generator=generator, num_inference_steps=refine_steps).images[0]
        print("Frame enhancement successful.")
    except Exception as e:
        print(f"Error during frame enhancement: {e}")
        gr.Warning("Frame enhancement failed. Using original frame.")
        return image_to_enhance
    finally:
        print("Moving enhancer pipeline to CPU...")
        enhancer_pipeline.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
    return enhanced_image

def use_last_frame_as_input(video_filepath, do_enhance, refine_prompt, refine_strength, refine_steps):
    if not video_filepath or not os.path.exists(video_filepath):
        gr.Warning("No video clip available.")
        return None, gr.update(), gr.update(value=False)
    cap = None
    try:
        cap = cv2.VideoCapture(video_filepath)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        if not ret: raise ValueError("Failed to read frame.")
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            pil_image.save(tmp_file.name)
            image_path = tmp_file.name

        print(f"Displaying original last frame from: {image_path}")
        yield image_path, gr.update(), gr.update(value=True)

        if do_enhance:
            enhanced_image = enhance_frame(pil_image, refine_prompt, refine_strength, refine_steps)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file_enhanced:
                enhanced_image.save(tmp_file_enhanced.name)
                enhanced_image_path = tmp_file_enhanced.name

            print(f"Displaying enhanced frame from {enhanced_image_path} and switching tab...")
            yield enhanced_image_path, gr.update(selected="i2v_tab"), gr.update(value=True)
        else:
            yield image_path, gr.update(selected="i2v_tab"), gr.update(value=True)
            
    except Exception as e:
        gr.Error(f"Failed to extract frame: {e}")
        return None, gr.update(), gr.update(value=False)
    finally:
        if cap: cap.release()


def stitch_videos(clips_list):
    if not clips_list or len(clips_list) < 2:
        raise gr.Error("You need at least two clips to stitch them together!")
    print(f"Stitching {len(clips_list)} clips...")
    try:
        video_clips = [VideoFileClip(clip_path) for clip_path in clips_list]
        final_clip = concatenate_videoclips(video_clips, method="compose")
        final_output_path = os.path.join(tempfile.mkdtemp(), f"stitched_video_{random.randint(10000,99999)}.mp4")
        final_clip.write_videofile(final_output_path, codec="libx264", audio=False, threads=4, preset='ultrafast')
        for clip in video_clips:
            clip.close()
        return final_output_path
    except Exception as e:
        raise gr.Error(f"Failed to stitch videos: {e}")


def clear_clips(latent_filepath):
    if latent_filepath and os.path.exists(latent_filepath):
        print(f"Deleting previous latent file: {latent_filepath}")
        try:
            os.remove(latent_filepath)
        except OSError as e:
            print(f"Error deleting latent file: {e}")
    return [], "Clips created: 0", None, None, None


@spaces.GPU(duration=get_duration)
def generate(prompt, negative_prompt, clips_list, previous_latent_path, continue_with_latents_flag,
             continuation_strength, mode, input_image_filepath, input_video_filepath, height_ui, width_ui, duration_ui,
             ui_frames_to_use, seed_ui, randomize_seed, ui_guidance_scale, improve_texture_flag,
             num_steps, fps, progress=gr.Progress(track_tqdm=True)):
    
    print("Clearing CUDA cache before generation...")
    gc.collect()
    torch.cuda.empty_cache()

    if mode not in ["text-to-video", "image-to-video", "video-to-video"]:
        raise gr.Error(f"Invalid mode: {mode}.")
    
    if mode == "image-to-video" and not input_image_filepath:
        raise gr.Error("input_image_filepath is required for image-to-video mode")
    elif mode == "video-to-video" and not input_video_filepath:
        raise gr.Error("input_video_filepath is required for video-to-video mode")
    
    if randomize_seed: seed_ui = random.randint(0, 2**32 - 1)
    seed_everething(int(seed_ui))

    actual_height, actual_width, height_padded, width_padded = 0, 0, 0, 0
    previous_latents = None
    skip_steps = 0

    if continue_with_latents_flag and previous_latent_path and os.path.exists(previous_latent_path):
        print(f"Loading previous latents from: {previous_latent_path}")
        previous_latents = torch.load(previous_latent_path)
        
        _, _, _, h_latent, w_latent = previous_latents.shape
        vae_scale_factor = pipeline_instance.vae_scale_factor
        height_padded = h_latent * vae_scale_factor
        width_padded = w_latent * vae_scale_factor
        actual_height = height_padded
        actual_width = width_padded
        print(f"Continue mode: Overriding UI dimensions. Using {actual_width}x{actual_height} from previous latent.")
        
        skip_steps = int(num_steps * (1.0 - continuation_strength))
        print(f"Continuation strength: {continuation_strength}, Total steps: {num_steps}, Skipping initial {skip_steps} steps.")

        print(f"Deleting used latent file: {previous_latent_path}")
        try:
            os.remove(previous_latent_path)
        except OSError as e:
            print(f"Error deleting used latent file: {e}")
    else:
        actual_height, actual_width = int(height_ui), int(width_ui)
        height_padded, width_padded = ((actual_height - 1) // 32 + 1) * 32, ((actual_width - 1) // 32 + 1) * 32

    actual_num_frames = max(9, min(MAX_NUM_FRAMES, int(round((max(1, round(duration_ui * fps)) - 1.0) / 8.0) * 8 + 1)))
    padding_values = calculate_padding(actual_height, actual_width, height_padded, width_padded)
    num_frames_padded = max(9, ((actual_num_frames - 2) // 8 + 1) * 8 + 1)
    generator_torch = torch.Generator(device=target_inference_device).manual_seed(int(seed_ui))
    
    call_kwargs = {
        "prompt": prompt, "negative_prompt": negative_prompt, "height": height_padded,
        "width": width_padded, "num_frames": num_frames_padded, "num_inference_steps": num_steps,
        "frame_rate": int(fps), "generator": generator_torch, "output_type": "latent",
        "conditioning_items": None, "media_items": None,
        "decode_timestep": PIPELINE_CONFIG_YAML["decode_timestep"], "decode_noise_scale": PIPELINE_CONFIG_YAML["decode_noise_scale"],
        "stochastic_sampling": PIPELINE_CONFIG_YAML["stochastic_sampling"], "image_cond_noise_scale": 0.15, "is_video": True,
        "vae_per_channel_normalize": True, "mixed_precision": (PIPELINE_CONFIG_YAML["precision"] == "mixed_precision"),
        "offload_to_cpu": False, "enhance_prompt": False,
        "skip_initial_inference_steps": skip_steps
    }
    
    stg_mode_str = PIPELINE_CONFIG_YAML.get("stg_mode", "attention_values").lower()
    stg_map = {"stg_av": SkipLayerStrategy.AttentionValues, "attention_values": SkipLayerStrategy.AttentionValues, "stg_as": SkipLayerStrategy.AttentionSkip, "attention_skip": SkipLayerStrategy.AttentionSkip, "stg_r": SkipLayerStrategy.Residual, "residual": SkipLayerStrategy.Residual, "stg_t": SkipLayerStrategy.TransformerBlock, "transformer_block": SkipLayerStrategy.TransformerBlock}
    call_kwargs["skip_layer_strategy"] = stg_map.get(stg_mode_str, SkipLayerStrategy.AttentionValues)

    if mode == "image-to-video":
        media_tensor = load_image_to_tensor_with_resize_and_crop(input_image_filepath, actual_height, actual_width)
        call_kwargs["conditioning_items"] = [ConditioningItem(torch.nn.functional.pad(media_tensor, padding_values).to(target_inference_device), 0, 1.0)]
    elif mode == "video-to-video": call_kwargs["media_items"] = load_media_file(media_path=input_video_filepath, height=actual_height, width=actual_width, max_frames=int(ui_frames_to_use), padding=padding_values).to(target_inference_device)

    if previous_latents is not None:
        previous_latents = previous_latents.to(target_inference_device)
        print(f"Splicing {N_LATENT_OVERLAP_FRAMES} latent frames from previous clip.")
        latent_shape = (1, pipeline_instance.vae.config.latent_channels, (num_frames_padded-1) // pipeline_instance.video_scale_factor + 1, height_padded // pipeline_instance.vae_scale_factor, width_padded // pipeline_instance.vae_scale_factor)
        initial_latents = randn_tensor(latent_shape, generator=generator_torch, device=target_inference_device, dtype=previous_latents.dtype)
        overlap_len = min(N_LATENT_OVERLAP_FRAMES, previous_latents.shape[2], initial_latents.shape[2])
        if overlap_len > 0:
            initial_latents[:, :, :overlap_len, :, :] = previous_latents[:, :, -overlap_len:, :, :]
        call_kwargs["latents"] = initial_latents
    
    pipeline_instance.to(target_inference_device)

    if improve_texture_flag and latent_upsampler_instance:
        multi_scale_pipeline = CorrectedLTXMultiScalePipeline(pipeline_instance, latent_upsampler_instance)
        multi_scale_kwargs = {"downscale_factor": PIPELINE_CONFIG_YAML["downscale_factor"], "first_pass": {**PIPELINE_CONFIG_YAML.get("first_pass", {})}, "second_pass": {**PIPELINE_CONFIG_YAML.get("second_pass", {})}}
        result_latents_tensor = multi_scale_pipeline(**call_kwargs, **multi_scale_kwargs).images
    else:
        single_pass_kwargs = {**call_kwargs, "guidance_scale": float(ui_guidance_scale), **PIPELINE_CONFIG_YAML.get("first_pass", {})}
        result_latents_tensor = pipeline_instance(**single_pass_kwargs).images
    
    print("Moving main transformer to CPU to free VRAM for VAE decoding...")
    pipeline_instance.transformer.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    if result_latents_tensor is None: raise gr.Error("Generation failed.")

    print("Decoding latents to video frames...")
    result_latents_tensor_for_decode = result_latents_tensor.to(pipeline_instance.vae.device, dtype=pipeline_instance.vae.dtype)
    decode_timestep_tensor = None
    if pipeline_instance.vae.decoder.timestep_conditioning:
        dt_value = call_kwargs["decode_timestep"]
        batch_size = result_latents_tensor_for_decode.shape[0]
        decode_timestep_tensor = torch.tensor([dt_value] * batch_size, device=result_latents_tensor_for_decode.device)
    
    print("Enabling VAE Z-Tiling for memory-efficient decoding.")
    pipeline_instance.vae.enable_z_tiling()
    pipeline_instance.vae.to(target_inference_device)
    
    result_images_tensor = vae_decode(result_latents_tensor_for_decode, pipeline_instance.vae, is_video=True, vae_per_channel_normalize=call_kwargs["vae_per_channel_normalize"], timestep=decode_timestep_tensor)
    
    print("Disabling VAE Z-Tiling.")
    pipeline_instance.vae.disable_z_tiling()
    
    result_images_tensor = (result_images_tensor + 1.0) / 2.0
    
    pad_l, pad_r, pad_t, pad_b = padding_values
    result_images_tensor = result_images_tensor[:, :, :actual_num_frames, pad_t:(-pad_b or None), pad_l:(-pad_r or None)]
    
    video_np = (np.clip(result_images_tensor[0].permute(1, 2, 3, 0).cpu().float().detach().numpy(), 0, 1) * 255).astype(np.uint8)
    
    output_video_path = os.path.join(tempfile.mkdtemp(), f"output_{random.randint(10000,99999)}.mp4")
    with imageio.get_writer(output_video_path, format='FFMPEG', fps=call_kwargs["frame_rate"], codec='libx264', quality=10, pixelformat='yuv420p') as video_writer:
        for idx, frame in enumerate(video_np):
            progress(idx / len(video_np), desc="Saving video clip...")
            video_writer.append_data(frame)
    updated_clips_list = clips_list + [output_video_path]
    counter_text = f"Clips created: {len(updated_clips_list)}"
    
    new_latent_path = None
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        new_latent_path = tmp_file.name
        print(f"Saving new latent tensor to: {new_latent_path}")
        torch.save(result_latents_tensor.cpu(), new_latent_path)

    return output_video_path, seed_ui, gr.update(visible=True), updated_clips_list, counter_text, new_latent_path


def update_task_image(): return "image-to-video"
def update_task_text(): return "text-to-video"
def update_task_video(): return "video-to-video"

css="""#col-container{margin:0 auto;max-width:900px;}"""

with gr.Blocks(css=css) as demo:
    clips_state = gr.State([])
    last_latent_state = gr.State(None)
    gr.Markdown("# LTX Video Clip Stitcher")
    gr.Markdown("Generate short video clips and stitch them together to create a longer animation.")
    with gr.Row():
        with gr.Column():
            with gr.Tabs() as tabs:
                with gr.Tab("image-to-video", id="i2v_tab") as image_tab:
                    video_i_hidden = gr.Textbox(visible=False);
                    image_i2v = gr.Image(label="Input Image", type="filepath", sources=["upload", "webcam", "clipboard"]);
                    i2v_prompt = gr.Textbox(label="Prompt", value="The creature from the image starts to move", lines=3);
                    i2v_button = gr.Button("Generate Image-to-Video Clip", variant="primary")
                with gr.Tab("text-to-video", id="t2v_tab") as text_tab:
                    image_n_hidden = gr.Textbox(visible=False);
                    video_n_hidden = gr.Textbox(visible=False); t2v_prompt = gr.Textbox(label="Prompt", value="A majestic dragon flying over a medieval castle", lines=3);
                    t2v_button = gr.Button("Generate Text-to-Video Clip", variant="primary")
                with gr.Tab("video-to-video", id="v2v_tab") as video_tab:
                    image_v_hidden = gr.Textbox(visible=False);
                    video_v2v = gr.Video(label="Input Video", sources=["upload", "webcam"]);
                    frames_to_use = gr.Slider(label="Frames to use from input video", minimum=9, maximum=120, value=9, step=8, info="Must be N*8+1.");
                    v2v_prompt = gr.Textbox(label="Prompt", value="Change the style to cinematic anime", lines=3);
                    v2v_button = gr.Button("Generate Video-to-Video Clip", variant="primary")
            duration_input = gr.Slider(label="Clip Duration (seconds)", minimum=1.0, maximum=10.0, value=2.0, step=0.1)
            improve_texture = gr.Checkbox(label="Improve Texture (multi-scale)", value=True)
            enhance_checkbox = gr.Checkbox(label="Enhance Last Frame with SDXL Refiner", value=True)
            with gr.Group(visible=True) as refiner_group:
                gr.Markdown("#### SDXL Refiner Settings")
                refiner_prompt_input = gr.Textbox(label="Refiner Prompt", value="4k, high-resolution, photorealistic, sharpened, intricate details", lines=2)
                refiner_strength_input = gr.Slider(label="Refiner Strength", minimum=0.0, maximum=0.5, step=0.01, value=0.15, info="How much to change the original frame. Lower is more faithful.")
                refiner_steps_input = gr.Slider(label="Refiner Steps", minimum=10, maximum=100, step=1, value=25)
        with gr.Column():
            output_video = gr.Video(label="Last Generated Clip", interactive=False)
            use_last_frame_button = gr.Button("Use Last Frame as Input Image", visible=False)
            with gr.Accordion("Stitching Controls", open=True):
                clip_counter_display = gr.Markdown("Clips created: 0")
                with gr.Row(): stitch_button = gr.Button("🎬 Stitch All Clips"); clear_button = gr.Button("🗑️ Clear All Clips")
                final_video_output = gr.Video(label="Final Stitched Video", interactive=False)
    with gr.Accordion("Advanced settings", open=False):
        mode = gr.Dropdown(["text-to-video", "image-to-video", "video-to-video"], label="task", value="image-to-video", visible=False);
        
        continue_with_latents = gr.Checkbox(label="Continue from previous clip's latents", value=False, info=f"Primes the new clip with the last {N_LATENT_OVERLAP_FRAMES*4} frames of the previous one.")
        with gr.Group(visible=False) as continuation_group:
            continuation_strength_input = gr.Slider(
                label="Continuation Denoising Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.75,
                info="How much to change the initial latent. 1.0=ignore previous clip, 0.0=no change."
            )
        
        negative_prompt_input = gr.Textbox(label="Negative Prompt", value="worst quality, inconsistent motion, blurry, jittery, distorted", lines=2)
        with gr.Row():
            seed_input = gr.Number(label="Seed", value=42, precision=0);
            randomize_seed_input = gr.Checkbox(label="Randomize Seed", value=True)
        with gr.Row(visible=False):
            guidance_scale_input = gr.Slider(label="Guidance Scale (CFG)", minimum=1.0, maximum=10.0, value=PIPELINE_CONFIG_YAML.get("first_pass", {}).get("guidance_scale", 1.0), step=0.1)
        with gr.Row():
            height_input = gr.Slider(label="Height", value=768, step=32, minimum=32, maximum=MAX_IMAGE_SIZE);
            width_input = gr.Slider(label="Width", value=768, step=32, minimum=32, maximum=MAX_IMAGE_SIZE);
        num_steps = gr.Slider(label="Steps", value=20, step=1, minimum=1, maximum=420);
        fps = gr.Slider(label="FPS", value=30.0, step=1.0, minimum=4.0, maximum=60.0)

    def handle_image_upload_for_dims(f, h, w):
        if not f: return gr.update(value=h), gr.update(value=w)
        img = Image.open(f); new_h, new_w = calculate_new_dimensions(img.width, img.height); return gr.update(value=new_h), gr.update(value=new_w)
    def handle_video_upload_for_dims(f, h, w):
        if not f or not os.path.exists(str(f)): return gr.update(value=h), gr.update(value=w)
        with imageio.get_reader(str(f)) as reader:
            meta = reader.get_meta_data(); orig_w, orig_h = meta.get('size', (reader.get_data(0).shape[1], reader.get_data(0).shape[0]));
            new_h, new_w = calculate_new_dimensions(orig_w, orig_h); return gr.update(value=new_h), gr.update(value=new_w)
    
    enhance_checkbox.change(fn=lambda x: gr.update(visible=x), inputs=enhance_checkbox, outputs=refiner_group)
    continue_with_latents.change(fn=lambda x: gr.update(visible=x), inputs=continue_with_latents, outputs=continuation_group)
    
    image_i2v.upload(handle_image_upload_for_dims, [image_i2v, height_input, width_input], [height_input, width_input]);
    video_v2v.upload(handle_video_upload_for_dims, [video_v2v, height_input, width_input], [height_input, width_input]);
    image_tab.select(update_task_image, outputs=[mode]); text_tab.select(update_task_text, outputs=[mode]);
    video_tab.select(update_task_video, outputs=[mode])
    
    common_params = [height_input, width_input, duration_input, frames_to_use, seed_input, randomize_seed_input, guidance_scale_input, improve_texture, num_steps, fps]
    t2v_inputs = [t2v_prompt, negative_prompt_input, clips_state, last_latent_state, continue_with_latents, continuation_strength_input, mode, image_n_hidden, video_n_hidden] + common_params;
    i2v_inputs = [i2v_prompt, negative_prompt_input, clips_state, last_latent_state, continue_with_latents, continuation_strength_input, mode, image_i2v, video_i_hidden] + common_params;
    v2v_inputs = [v2v_prompt, negative_prompt_input, clips_state, last_latent_state, continue_with_latents, continuation_strength_input, mode, image_v_hidden, video_v2v] + common_params
    
    gen_outputs = [output_video, seed_input, use_last_frame_button, clips_state, clip_counter_display, last_latent_state]
    hide_btn = lambda: gr.update(visible=False)
    t2v_button.click(hide_btn, outputs=[use_last_frame_button], queue=False).then(fn=generate, inputs=t2v_inputs, outputs=gen_outputs, api_name="text_to_video")
    i2v_button.click(hide_btn, outputs=[use_last_frame_button], queue=False).then(fn=generate, inputs=i2v_inputs, outputs=gen_outputs, api_name="image_to_video")
    v2v_button.click(hide_btn, outputs=[use_last_frame_button], queue=False).then(fn=generate, inputs=v2v_inputs, outputs=gen_outputs, api_name="video_to_video")
    use_last_frame_button.click(fn=use_last_frame_as_input, inputs=[output_video, enhance_checkbox, refiner_prompt_input, refiner_strength_input, refiner_steps_input], outputs=[image_i2v, tabs, continue_with_latents])
    stitch_button.click(fn=stitch_videos, inputs=[clips_state], outputs=[final_video_output])
    clear_button.click(fn=clear_clips, inputs=[last_latent_state], outputs=[clips_state, clip_counter_display, output_video, final_video_output, last_latent_state])

if __name__ == "__main__":
    if os.path.exists(models_dir): print(f"Model directory: {Path(models_dir).resolve()}")
    demo.queue().launch(debug=True, share=False, mcp_server=True)
