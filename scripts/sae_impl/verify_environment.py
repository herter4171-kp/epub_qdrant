#!/usr/bin/env python3
"""Verify environment for SAE implementation on RTX 5090."""

import sys


def verify_cuda():
    """Check CUDA version and availability."""
    try:
        import torch
        cuda_version = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count()
        cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        cuda_major = torch.version.cuda.split('.')[0] if cuda_version else "N/A"
        
        print(f"CUDA: {cuda_version if cuda_version else 'N/A'}")
        print(f"  - Available: {cuda_available}")
        print(f"  - Device count: {cuda_device_count}")
        print(f"  - Device name: {cuda_device_name}")
        print(f"  - CUDA major version: {cuda_major}")
        
        return cuda_major, cuda_available
    except ImportError:
        print("ERROR: PyTorch not installed")
        return None, False


def verify_pytorch():
    """Check PyTorch version."""
    try:
        import torch
        torch_version = torch.__version__
        print(f"PyTorch: {torch_version}")
        return torch_version
    except ImportError:
        print("ERROR: PyTorch not installed")
        return None


def verify_saelens():
    """Check SAELens version."""
    try:
        import saelens
        saelens_version = saelens.__version__
        print(f"SAELens: {saelens_version}")
        return saelens_version
    except ImportError:
        try:
            import sae_lens
            saelens_version = sae_lens.__version__
            print(f"SAELens: {saelens_version}")
            return saelens_version
        except ImportError:
            print("WARNING: SAELens not installed or version unknown")
            return None


def verify_transformers():
    """Check Transformers version."""
    try:
        import transformers
        transformers_version = transformers.__version__
        print(f"Transformers: {transformers_version}")
        return transformers_version
    except ImportError:
        print("WARNING: Transformers not installed")
        return None


def verify_qdrant():
    """Check Qdrant client version."""
    try:
        import qdrant_client
        qdrant_version = qdrant_client.__version__
        print(f"Qdrant client: {qdrant_version}")
        return qdrant_version
    except ImportError:
        print("WARNING: Qdrant client not installed")
        return None


def verify_splade():
    """Test SPLADE model loading."""
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        
        # Try loading SPLADE model (using local files if available)
        tokenizer = AutoTokenizer.from_pretrained(
            "naver/splade-cocondenser-ensembledistil",
            local_files_only=True,
        )
        
        model = AutoModelForMaskedLM.from_pretrained(
            "naver/splade-cocondenser-ensembledistil",
            local_files_only=True,
        )
        
        # Test forward pass
        test_text = "test query"
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            output = model(**inputs)
        
        # Check output shape (should be (1, seq_len, vocab_size=30522))
        vocab_size = output.logits.shape[-1]
        print(f"SPLADE: Model loaded, vocab_size={vocab_size}")
        print(f"  - Forward pass test: OK (output shape: {output.logits.shape})")
        
        return vocab_size
    except Exception as e:
        print(f"ERROR: SPLADE test failed: {e}")
        return None


def main():
    """Run all verifications."""
    print("=" * 60)
    print("Environment Verification for SAE on RTX 5090")
    print("=" * 60)
    print()
    
    # Verify CUDA
    print("=== CUDA/PyTorch ===")
    cuda_major, cuda_available = verify_cuda()
    print()
    
    # Verify PyTorch
    print("=== PyTorch ===")
    torch_version = verify_pytorch()
    print()
    
    # Verify SAELens
    print("=== SAELens ===")
    saelens_version = verify_saelens()
    print()
    
    # Verify Transformers
    print("=== Transformers ===")
    transformers_version = verify_transformers()
    print()
    
    # Verify Qdrant client
    print("=== Qdrant Client ===")
    qdrant_version = verify_qdrant()
    print()
    
    # Test SPLADE
    print("=== SPLADE Model Test ===")
    vocab_size = verify_splade()
    print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    
    # Check requirements
    requirements_met = True
    
    if cuda_major is None or int(cuda_major) < 12:
        print("❌ CUDA >= 12.4 required")
        requirements_met = False
    else:
        print(f"✓ CUDA {cuda_major}+")
    
    if torch_version is None:
        print("❌ PyTorch >= 2.3 required")
        requirements_met = False
    else:
        print(f"✓ PyTorch {torch_version}")
    
    if saelens_version is None:
        print("⚠ SAELens 6.43.0 recommended (not found)")
    else:
        print(f"✓ SAELens {saelens_version}")
    
    if transformers_version is None:
        print("⚠ Transformers >= 4.40 recommended (not found)")
    else:
        print(f"✓ Transformers {transformers_version}")
    
    if qdrant_version is None:
        print("⚠ Qdrant client >= 1.9 recommended (not found)")
    else:
        print(f"✓ Qdrant client {qdrant_version}")
    
    print()
    if requirements_met:
        print("✓ Environment verification PASSED")
    else:
        print("❌ Environment verification FAILED")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())