"""
Remote Text Encoder Client
Utility for calling remote text encoder spaces via Gradio Client API.
"""

import torch
import os
from typing import Optional, Tuple, Dict, Any
import io
import pickle


class RemoteTextEncoderClient:
    """Client for remote text encoding via Gradio API."""
    
    def __init__(self, encoder_space_url: Optional[str] = None):
        """
        Initialize remote text encoder client.
        
        Args:
            encoder_space_url: URL of the encoder space (e.g., "username/space-name" or full URL)
                              If None, will try to get from REMOTE_ENCODER_SPACE_URL env var
        """
        self.encoder_space_url = encoder_space_url or os.getenv("REMOTE_ENCODER_SPACE_URL")
        self.client = None
        self._initialized = False
        
    def _init_client(self):
        """Lazy initialization of Gradio client."""
        if self._initialized:
            return
            
        if not self.encoder_space_url:
            raise ValueError(
                "No encoder space URL provided. Set encoder_space_url or REMOTE_ENCODER_SPACE_URL env var."
            )
        
        try:
            from gradio_client import Client
            print(f"Connecting to remote encoder at {self.encoder_space_url}...")
            self.client = Client(self.encoder_space_url)
            self._initialized = True
            print("✅ Connected to remote encoder successfully")
        except Exception as e:
            print(f"❌ Failed to connect to remote encoder: {e}")
            raise
    
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode a prompt using remote text encoder.
        
        Args:
            prompt: Text prompt to encode
            negative_prompt: Optional negative prompt
            device: Device to move tensors to after receiving
            
        Returns:
            Tuple of (video_context, audio_context, video_context_negative, audio_context_negative)
            Negative contexts will be None if no negative_prompt provided
        """
        self._init_client()
        
        try:
            print(f"🔄 Encoding prompt remotely: '{prompt[:50]}...'")
            result = self.client.predict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                api_name="/encode_api"
            )
            
            # Result is a tuple: (embedding_data, status)
            embedding_data, status = result
            print(f"✅ Remote encoding complete: {status}")
            
            # Convert embedding data back to tensors with proper shapes
            video_context = self._deserialize_tensor(
                embedding_data['video_context'],
                embedding_data['video_context_shape'],
                device
            )
            audio_context = self._deserialize_tensor(
                embedding_data['audio_context'],
                embedding_data['audio_context_shape'],
                device
            )
            
            video_context_negative = None
            audio_context_negative = None
            if 'video_context_negative' in embedding_data:
                video_context_negative = self._deserialize_tensor(
                    embedding_data['video_context_negative'],
                    embedding_data['video_context_negative_shape'],
                    device
                )
                audio_context_negative = self._deserialize_tensor(
                    embedding_data['audio_context_negative'],
                    embedding_data['audio_context_negative_shape'],
                    device
                )
            
            return video_context, audio_context, video_context_negative, audio_context_negative
            
        except Exception as e:
            print(f"❌ Remote encoding failed: {e}")
            raise
    
    def _deserialize_tensor(self, tensor_data: Any, tensor_shape: list, device: str) -> torch.Tensor:
        """Convert received tensor data to torch.Tensor on specified device."""
        if isinstance(tensor_data, torch.Tensor):
            return tensor_data.to(device)
        
        # Convert list to tensor with the provided shape
        if isinstance(tensor_data, (list, tuple)):
            tensor = torch.tensor(tensor_data).reshape(tensor_shape)
            return tensor.to(device)
        
        # If it's already a numpy array
        if hasattr(tensor_data, 'shape'):
            tensor = torch.from_numpy(tensor_data)
            return tensor.to(device)
        
        # Try direct conversion as fallback
        try:
            tensor = torch.tensor(tensor_data)
            return tensor.to(device)
        except Exception as e:
            print(f"Warning: Failed to deserialize tensor: {e}")
            raise


def encode_prompt_with_remote_fallback(
    prompt: str,
    negative_prompt: str = "",
    remote_encoder_url: Optional[str] = None,
    local_encoder_fn: Optional[callable] = None,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Encode prompt with remote encoder, falling back to local if remote fails.
    
    Args:
        prompt: Text prompt to encode
        negative_prompt: Optional negative prompt
        remote_encoder_url: URL of remote encoder space (or use env var)
        local_encoder_fn: Fallback local encoding function
        device: Device to use
        
    Returns:
        Tuple of (video_context, audio_context, video_context_negative, audio_context_negative)
    """
    # Try remote encoding first if URL is provided
    if remote_encoder_url or os.getenv("REMOTE_ENCODER_SPACE_URL"):
        try:
            client = RemoteTextEncoderClient(remote_encoder_url)
            return client.encode_prompt(prompt, negative_prompt, device)
        except Exception as e:
            print(f"⚠️ Remote encoding failed, trying fallback: {e}")
    
    # Fall back to local encoding
    if local_encoder_fn:
        print("🔄 Using local text encoder")
        return local_encoder_fn(prompt, negative_prompt)
    
    raise ValueError("No encoding method available (neither remote nor local)")
