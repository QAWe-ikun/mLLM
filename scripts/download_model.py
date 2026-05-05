#!/usr/bin/env python
"""
脚本: 下载预训练模型到本地 (使用 ModelScope 国内镜像)

用法:
    python scripts/download_model.py                          # 下载全部模型
    python scripts/download_model.py --only-qwen              # 只下载 Qwen3-VL
    python scripts/download_model.py --only-clip              # 只下载 CLIP
    python scripts/download_model.py --force                  # 强制重新下载
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path


def check_modelscope():
    """检查/安装 ModelScope"""
    try:
        import modelscope
        print(f"[OK] ModelScope 已安装: {modelscope.__version__}")
        return True
    except ImportError:
        print("[!] ModelScope 未找到，正在安装...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope", "--upgrade"])
        print("[OK] ModelScope 安装完成")
        return True


def is_model_downloaded(local_dir: str, check_file: str = "config.json") -> bool:
    """检查模型是否已下载完成
    
    Args:
        local_dir: 模型保存路径
        check_file: 用于验证的关键文件
        
    Returns:
        是否已下载
    """
    path = Path(local_dir)
    
    # 目录不存在
    if not path.exists():
        return False
    
    # 检查关键文件
    config_path = path / check_file
    if config_path.exists():
        return True
    
    # 检查子目录
    sub_configs = list(path.glob(f"**/{check_file}"))
    if sub_configs:
        return True
    
    return False


def download_model(model_id: str, local_dir: str, force: bool = False, 
                   source_name: str = "ModelScope") -> str:
    """下载单个模型
    
    Args:
        model_id: ModelScope 上的模型ID
        local_dir: 本地保存路径
        force: 是否强制重新下载
        source_name: 来源名称（用于日志）
        
    Returns:
        实际下载路径
    """
    print(f"\n{'='*60}")
    print(f"模型: {model_id}")
    print(f"来源: {source_name}")
    print(f"目标: {local_dir}")
    print(f"{'='*60}")
    
    # 检查是否已存在
    if not force and is_model_downloaded(local_dir):
        print(f"[OK] 模型已存在: {local_dir}")
        print("     跳过下载 (使用 --force 强制重新下载)")
        return local_dir
    
    try:
        from modelscope import snapshot_download
        
        print(f"开始下载...")
        result_dir = snapshot_download(
            model_id,
            local_dir=local_dir,
            revision='master'
        )
        
        print(f"[OK] 下载完成: {result_dir}")
        return result_dir
        
    except Exception as e:
        print(f"\n[ERROR] 下载失败: {e}")
        print("提示: 检查网络连接，或重新运行脚本可断点续传")
        raise


def main():
    parser = argparse.ArgumentParser(description='下载预训练模型 (ModelScope)')
    parser.add_argument('--force', action='store_true',
                       help='强制重新下载')
    parser.add_argument('--only-qwen', action='store_true',
                       help='只下载 Qwen3-VL')
    parser.add_argument('--only-clip', action='store_true',
                       help='只下载 CLIP')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='模型保存根目录 (默认: models)')
    
    args = parser.parse_args()
    
    print("VLA Agent 模型下载工具 (ModelScope)")
    print("="*60)
    
    # 检查依赖
    if not check_modelscope():
        print("[ERROR] ModelScope 安装失败")
        sys.exit(1)
    
    # 创建目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    try:
        # 下载 Qwen3-VL 8B
        if not args.only_clip:
            print("\n>>> 步骤 1/2: Qwen3-VL-8B")
            results['qwen'] = download_model(
                model_id='Qwen/Qwen3-VL-8B-Instruct',
                local_dir=str(output_dir / 'Qwen3-VL-8B'),
                force=args.force,
            )
        
        # 下载 CLIP ViT-B/32
        if not args.only_qwen:
            print("\n>>> 步骤 2/2: CLIP ViT-B/32")
            results['clip'] = download_model(
                model_id='AI-ModelScope/clip-vit-base-patch32',
                local_dir=str(output_dir / 'clip-vit-base-patch32'),
                force=args.force,
            )
        
        # 汇总
        print("\n" + "="*60)
        print("下载汇总")
        print("="*60)
        for name, path in results.items():
            print(f"  {name}: {path}")
        
        print(f"\n模型已保存到 {output_dir}/ 目录")
        print("可以直接运行训练脚本了！")
        
    except Exception as e:
        print(f"\n[ERROR] 下载失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
