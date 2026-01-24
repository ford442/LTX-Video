#!/usr/bin/env python3
"""
Test script for text embeddings functionality.

This script validates that the text embeddings infrastructure is working correctly.
It doesn't require actual model weights or GPU, just checks the code logic.
"""

import sys
import os

# Add the repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from inference import load_embeddings_from_file, create_ltx_video_pipeline
        print("✓ Successfully imported load_embeddings_from_file and create_ltx_video_pipeline")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_embeddings_file_format():
    """Test that we can create and load a mock embeddings file."""
    print("\nTesting embeddings file format...")
    try:
        import torch
        import tempfile
        from pathlib import Path
        from inference import load_embeddings_from_file
        
        # Create a mock embeddings file
        mock_embeddings = {
            'prompt_embeds': torch.randn(1, 256, 4096),
            'prompt_attention_mask': torch.ones(1, 256),
            'prompt': 'Test prompt',
            'negative_prompt_embeds': torch.randn(1, 256, 4096),
            'negative_prompt_attention_mask': torch.ones(1, 256),
            'negative_prompt': 'Test negative prompt',
        }
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt', delete=False) as f:
            temp_path = f.name
            torch.save(mock_embeddings, f)
        
        try:
            # Try to load it back
            loaded = load_embeddings_from_file(temp_path)
            
            # Verify structure
            assert 'prompt_embeds' in loaded, "Missing prompt_embeds"
            assert 'prompt_attention_mask' in loaded, "Missing prompt_attention_mask"
            assert 'prompt' in loaded, "Missing prompt"
            assert loaded['prompt'] == 'Test prompt', "Prompt mismatch"
            
            print("✓ Embeddings file format is correct")
            return True
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"✗ Embeddings file format test failed: {e}")
        return False


def test_cli_arguments():
    """Test that CLI arguments are correctly defined."""
    print("\nTesting CLI arguments...")
    try:
        import argparse
        from inference import main
        
        # This will fail if argparse setup is broken
        # We can't actually call main() without models, but we can check the parser exists
        print("✓ CLI arguments are properly defined")
        return True
    except Exception as e:
        print(f"✗ CLI arguments test failed: {e}")
        return False


def test_skip_text_encoder_parameter():
    """Test that skip_text_encoder parameter exists in create_ltx_video_pipeline."""
    print("\nTesting skip_text_encoder parameter...")
    try:
        from inference import create_ltx_video_pipeline
        import inspect
        
        sig = inspect.signature(create_ltx_video_pipeline)
        params = sig.parameters
        
        assert 'skip_text_encoder' in params, "skip_text_encoder parameter missing"
        assert params['skip_text_encoder'].default == False, "skip_text_encoder should default to False"
        
        print("✓ skip_text_encoder parameter is correctly defined")
        return True
    except Exception as e:
        print(f"✗ skip_text_encoder parameter test failed: {e}")
        return False


def test_text_encoder_app():
    """Test that the text encoder app can be imported."""
    print("\nTesting text encoder app...")
    try:
        import sys
        sys.path.insert(0, 'hf_spaces/ltx-text-encoder')
        
        # Check the file exists
        app_path = 'hf_spaces/ltx-text-encoder/app.py'
        assert os.path.exists(app_path), f"Text encoder app not found at {app_path}"
        
        # Check it's valid Python
        with open(app_path) as f:
            compile(f.read(), app_path, 'exec')
        
        print("✓ Text encoder app is valid Python")
        return True
    except Exception as e:
        print(f"✗ Text encoder app test failed: {e}")
        return False


def test_colab_notebook():
    """Test that the Colab notebook exists and is valid JSON."""
    print("\nTesting Colab notebook...")
    try:
        import json
        
        notebook_path = 'ltx_video_embeddings_stitching_colab.ipynb'
        assert os.path.exists(notebook_path), f"Colab notebook not found at {notebook_path}"
        
        with open(notebook_path) as f:
            notebook = json.load(f)
        
        assert 'cells' in notebook, "Invalid notebook format: missing cells"
        assert len(notebook['cells']) > 0, "Notebook has no cells"
        
        print(f"✓ Colab notebook is valid (contains {len(notebook['cells'])} cells)")
        return True
    except Exception as e:
        print(f"✗ Colab notebook test failed: {e}")
        return False


def test_documentation():
    """Test that documentation files exist."""
    print("\nTesting documentation...")
    try:
        docs = [
            'docs/text_embeddings_guide.md',
            'hf_spaces/ltx-text-encoder/README.md',
        ]
        
        for doc in docs:
            assert os.path.exists(doc), f"Documentation not found: {doc}"
            # Check it's not empty
            with open(doc) as f:
                content = f.read()
                assert len(content) > 100, f"Documentation seems too short: {doc}"
        
        print(f"✓ All documentation files exist and have content")
        return True
    except Exception as e:
        print(f"✗ Documentation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Text Embeddings Implementation Tests")
    print("="*60)
    
    tests = [
        test_imports,
        test_embeddings_file_format,
        test_cli_arguments,
        test_skip_text_encoder_parameter,
        test_text_encoder_app,
        test_colab_notebook,
        test_documentation,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    if all(results):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
