#!/usr/bin/env python
"""
脚本: 下载预训练模型到本地

用法:
    python scripts/download_model.py --model Qwen/Qwen3-VL-8B-Instruct --output models/Qwen3-VL-8B
    python scripts/download_model.py --model openai/clip-vit-base-patch32 --output models/clip-vit-b32
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path


def download_model(model_name: str, output_dir: str, use_mirror: bool = True):
    """下载模型到本地
    
    Args:
        model_name: HuggingFace模型ID
        output_dir: 本地保存路径
        use_mirror: 是否使用HF镜像加速(国内推荐)
    """
    import os
    
    if use_mirror:
        os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Downloading model: {model_name}")
    print(f"Save to: {output_path}")
    print(f"Using mirror: {use_mirror}")
    print(f"=" * 60)
    
    # 下载 tokenizer
    print("\n[1/2] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=str(output_path),
    )
    tokenizer.save_pretrained(str(output_path))
    print(f"Tokenizer saved to {output_path}")
    
    # 下载模型
    print("\n[2/2] Downloading model weights...")
    print("This may take a while depending on your network speed...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=str(output_path),
    )
    model.save_pretrained(str(output_path))
    print(f"Model saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("Download completed!")
    print(f"Model path: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Download model from HuggingFace')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct',
                       help='Model ID on HuggingFace')
    parser.add_argument('--output', type=str, default='models/Qwen3-VL-8B',
                       help='Local save path')
    parser.add_argument('--no-mirror', action='store_true',
                       help='Disable HuggingFace mirror')
    parser.add_argument('--clip', action='store_true',
                       help='Download CLIP model instead')
    
    args = parser.parse_args()
    
    if args.clip:
        # 下载CLIP
        print("Downloading CLIP model...")
        from transformers import CLIPVisionModel, CLIPImageProcessor
        model_name = "openai/clip-vit-base-patch32"
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processor = CLIPImageProcessor.from_pretrained(model_name)
        processor.save_pretrained(str(output_path))
        
        model = CLIPVisionModel.from_pretrained(model_name)
        model.save_pretrained(str(output_path))
        
        print(f"CLIP model saved to {output_path}")
    else:
        download_model(
            model_name=args.model,
            output_dir=args.output,
            use_mirror=not args.no_mirror,
        )


if __name__ == '__main__':
    main()
