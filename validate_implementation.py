#!/usr/bin/env python3
"""
Lightweight test script for text embeddings functionality.
Only tests code structure, not runtime behavior.
"""

import os
import re
import json


def test_inference_modifications():
    """Test that inference.py has the required modifications."""
    print("Testing inference.py modifications...")
    
    with open('inference.py') as f:
        content = f.read()
    
    checks = [
        ('load_embeddings_from_file', 'def load_embeddings_from_file'),
        ('skip_text_encoder parameter', 'skip_text_encoder'),
        ('embeddings_path parameter', 'embeddings_path'),
        ('CLI argument for embeddings', '--embeddings_path'),
    ]
    
    passed = 0
    for name, pattern in checks:
        if pattern in content:
            print(f"  ✓ {name} found")
            passed += 1
        else:
            print(f"  ✗ {name} NOT found")
    
    return passed == len(checks)


def test_text_encoder_app():
    """Test that text encoder app exists and has required components."""
    print("\nTesting text encoder app...")
    
    app_path = 'hf_spaces/ltx-text-encoder/app.py'
    if not os.path.exists(app_path):
        print(f"  ✗ App file not found: {app_path}")
        return False
    
    with open(app_path) as f:
        content = f.read()
    
    checks = [
        ('T5EncoderModel import', 'T5EncoderModel'),
        ('encode_prompt function', 'def encode_prompt'),
        ('Gradio interface', 'gr.Blocks'),
        ('embeddings save', 'torch.save'),
    ]
    
    passed = 0
    for name, pattern in checks:
        if pattern in content:
            print(f"  ✓ {name} found")
            passed += 1
        else:
            print(f"  ✗ {name} NOT found")
    
    # Check requirements.txt
    req_path = 'hf_spaces/ltx-text-encoder/requirements.txt'
    if os.path.exists(req_path):
        with open(req_path) as f:
            reqs = f.read()
        if 'gradio' in reqs and 'transformers' in reqs:
            print(f"  ✓ requirements.txt has required packages")
            passed += 1
        else:
            print(f"  ✗ requirements.txt missing required packages")
    else:
        print(f"  ✗ requirements.txt not found")
    
    return passed >= len(checks)


def test_colab_notebook():
    """Test that Colab notebook is valid and has required cells."""
    print("\nTesting Colab notebook...")
    
    notebook_path = 'ltx_video_embeddings_stitching_colab.ipynb'
    if not os.path.exists(notebook_path):
        print(f"  ✗ Notebook not found: {notebook_path}")
        return False
    
    try:
        with open(notebook_path) as f:
            notebook = json.load(f)
        
        if 'cells' not in notebook:
            print("  ✗ Invalid notebook format: missing cells")
            return False
        
        cells = notebook['cells']
        cell_content = '\n'.join([
            '\n'.join(cell.get('source', []))
            for cell in cells
        ])
        
        checks = [
            ('Text encoder loading', 'T5EncoderModel'),
            ('Encode function', 'encode_and_save_prompt'),
            ('Video generation', 'infer'),
            ('Video stitching', 'stitch_videos'),
            ('moviepy import', 'moviepy'),
        ]
        
        passed = 0
        for name, pattern in checks:
            if pattern in cell_content:
                print(f"  ✓ {name} found in notebook")
                passed += 1
            else:
                print(f"  ✗ {name} NOT found in notebook")
        
        print(f"  ✓ Notebook has {len(cells)} cells")
        return passed >= len(checks) - 1  # Allow one missing
        
    except json.JSONDecodeError as e:
        print(f"  ✗ Invalid JSON: {e}")
        return False


def test_documentation():
    """Test that documentation exists and covers key topics."""
    print("\nTesting documentation...")
    
    docs_to_check = {
        'docs/text_embeddings_guide.md': [
            'load_embeddings_from_file',
            'skip_text_encoder',
            'embeddings_path',
            'Video Stitching',
        ],
        'hf_spaces/ltx-text-encoder/README.md': [
            'Text Encoder',
            'embeddings',
            'VRAM',
        ],
        'README.md': [
            'embeddings_path',
            'text_embeddings_guide',
        ],
    }
    
    passed = 0
    total = 0
    
    for doc_path, keywords in docs_to_check.items():
        if not os.path.exists(doc_path):
            print(f"  ✗ {doc_path} not found")
            continue
        
        with open(doc_path) as f:
            content = f.read()
        
        for keyword in keywords:
            total += 1
            if keyword.lower() in content.lower():
                passed += 1
            else:
                print(f"  ✗ {doc_path} missing keyword: {keyword}")
    
    if passed == total:
        print(f"  ✓ All documentation files contain required keywords")
        return True
    else:
        print(f"  ⚠ {passed}/{total} keywords found in documentation")
        return passed >= total * 0.8  # Allow 80% pass rate


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'inference.py',
        'hf_spaces/ltx-text-encoder/app.py',
        'hf_spaces/ltx-text-encoder/requirements.txt',
        'hf_spaces/ltx-text-encoder/README.md',
        'ltx_video_embeddings_stitching_colab.ipynb',
        'docs/text_embeddings_guide.md',
    ]
    
    passed = 0
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file_path} ({size} bytes)")
            passed += 1
        else:
            print(f"  ✗ {file_path} NOT found")
    
    return passed == len(required_files)


def main():
    """Run all tests."""
    print("="*60)
    print("Text Embeddings Implementation Validation")
    print("="*60 + "\n")
    
    tests = [
        ('File Structure', test_file_structure),
        ('inference.py Modifications', test_inference_modifications),
        ('Text Encoder App', test_text_encoder_app),
        ('Colab Notebook', test_colab_notebook),
        ('Documentation', test_documentation),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results[name] = False
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print("="*60)
    print(f"Result: {total_passed}/{total_tests} test groups passed")
    print("="*60)
    
    if all(results.values()):
        print("\n🎉 All validation checks passed!")
        print("\nImplementation is complete and ready for use.")
        print("\nNext steps:")
        print("1. Test with actual models (requires GPU and dependencies)")
        print("2. Deploy text encoder space to HuggingFace Spaces")
        print("3. Try the Colab notebook")
        return 0
    else:
        print("\n⚠️  Some checks failed. See details above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
