#!/usr/bin/env python3
"""
Test script for the remote text encoder API.
This script tests the ltx-2-text-encoder space API functionality.
"""

import sys
import os

# Add the hf_spaces directory to path so we can import remote_text_encoder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hf_spaces'))

def test_encode_prompt_api():
    """Test the encode_prompt_api function directly."""
    print("=" * 80)
    print("Testing encode_prompt_api function")
    print("=" * 80)
    
    # This test requires the ltx-2-text-encoder to be loaded
    # In a real deployment, this would be tested via the deployed space
    
    try:
        # Import the app module (this will load the model)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hf_spaces/ltx-2-text-encoder'))
        from app import encode_prompt_api
        
        print("\n📝 Testing basic prompt encoding...")
        prompt = "A cat sitting on a windowsill"
        negative_prompt = "blurry, low quality"
        
        embeddings, status = encode_prompt_api(prompt, negative_prompt)
        
        if embeddings is None:
            print(f"❌ Encoding failed: {status}")
            return False
        
        print(f"✅ {status}")
        print(f"\n📊 Embedding shapes:")
        print(f"   video_context: {embeddings['video_context_shape']}")
        print(f"   audio_context: {embeddings['audio_context_shape']}")
        
        if 'video_context_negative_shape' in embeddings:
            print(f"   video_context_negative: {embeddings['video_context_negative_shape']}")
            print(f"   audio_context_negative: {embeddings['audio_context_negative_shape']}")
        
        print("\n✅ API function test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ API function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_remote_client():
    """Test the RemoteTextEncoderClient (requires deployed space)."""
    print("\n" + "=" * 80)
    print("Testing RemoteTextEncoderClient")
    print("=" * 80)
    
    encoder_url = os.getenv("REMOTE_ENCODER_SPACE_URL")
    
    if not encoder_url:
        print("⚠️  Skipping remote client test - REMOTE_ENCODER_SPACE_URL not set")
        print("   To test remote encoding, set REMOTE_ENCODER_SPACE_URL to your deployed space")
        print("   Example: export REMOTE_ENCODER_SPACE_URL=username/ltx-2-text-encoder")
        return None
    
    try:
        from remote_text_encoder import RemoteTextEncoderClient
        import torch
        
        print(f"\n🔌 Connecting to remote encoder at: {encoder_url}")
        client = RemoteTextEncoderClient(encoder_url)
        
        print("\n📝 Testing remote encoding...")
        prompt = "A dog running through a field"
        negative_prompt = "blurry, distorted"
        
        video_ctx, audio_ctx, video_neg, audio_neg = client.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device="cpu"  # Use CPU for testing
        )
        
        print(f"\n✅ Remote encoding successful!")
        print(f"📊 Embedding shapes:")
        print(f"   video_context: {video_ctx.shape}")
        print(f"   audio_context: {audio_ctx.shape}")
        
        if video_neg is not None:
            print(f"   video_context_negative: {video_neg.shape}")
            print(f"   audio_context_negative: {audio_neg.shape}")
        
        print("\n✅ Remote client test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Remote client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("REMOTE TEXT ENCODER TEST SUITE")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Direct API function
    print("\n[Test 1/2] Testing encode_prompt_api function...")
    results['api_function'] = test_encode_prompt_api()
    
    # Test 2: Remote client
    print("\n[Test 2/2] Testing RemoteTextEncoderClient...")
    results['remote_client'] = test_remote_client()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        
        print(f"{test_name:20s}: {status}")
    
    # Exit with appropriate code
    if any(r is False for r in results.values()):
        print("\n❌ Some tests failed")
        sys.exit(1)
    elif all(r is True for r in results.values()):
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests skipped")
        sys.exit(0)


if __name__ == "__main__":
    main()
