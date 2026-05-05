#!/usr/bin/env python
"""
脚本: 下载预训练模型到本地 (全部使用 ModelScope 国内镜像)

策略:
- Qwen3-VL: 'Qwen/Qwen3-VL-8B-Instruct'
- CLIP:     'openai-mirror/clip-vit-base-patch32'

用法:
    python scripts/download_model.py                          # 下载全部模型
    python scripts/download_model.py --only-qwen              # 只下载 Qwen3-VL
    python scripts/download_model.py --only-clip              # 只下载 CLIP
    python scripts/download_model.py --force                  # 强制重新下载
"""
import sys
import os
import argparse
import shutil
from pathlib import Path


def clean_modelscope_temp_files(directory: str):
    """清理 ModelScope 产生的临时文件和锁
    
    下载完成后，ModelScope 可能会留下 `____temp` 目录和 `.lock` 文件。
    此函数会递归扫描并删除它们，以节省空间。
    
    Args:
        directory: 需要清理的根目录
    """
    if not os.path.exists(directory):
        return

    cleaned_count = 0

    for root, dirs, files in os.walk(directory):
        # 删除 .____temp 目录
        for dir_name in dirs:
            if dir_name == '.____temp':
                temp_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(temp_path)
                    print(f"  [清理] 删除临时目录: {temp_path}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"  [警告] 无法删除 {temp_path}: {e}")

        # 删除 .lock 文件
        for file_name in files:
            if file_name.endswith('.lock'):
                lock_path = os.path.join(root, file_name)
                try:
                    os.remove(lock_path)
                    print(f"  [清理] 删除锁文件: {lock_path}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"  [警告] 无法删除 {lock_path}: {e}")

    if cleaned_count > 0:
        print(f"[OK] 清理了 {cleaned_count} 个临时项: {directory}")


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
    """检查模型是否已下载"""
    path = Path(local_dir)
    if not path.exists():
        return False
    if (path / check_file).exists():
        return True
    if list(path.glob(f"**/{check_file}")):
        return True
    return False


def download_model(model_id: str, local_dir: str, force: bool = False) -> str:
    """使用 ModelScope 下载模型

    Args:
        model_id: ModelScope 模型 ID
        local_dir: 本地保存路径
        force: 强制重下
    """
    print(f"\n{'='*60}")
    print(f"模型: {model_id}")
    print(f"目标: {local_dir}")
    print(f"{'='*60}")

    if not force and is_model_downloaded(local_dir):
        # 即使已下载，也检查一下是否有残留的临时文件
        print(f"[OK] 模型已存在: {local_dir}")
        return local_dir

    try:
        from modelscope import snapshot_download

        print(f"开始下载 (支持断点续传)...")
        
        # ModelScope snapshot_download 支持 cache_dir 指定缓存位置
        result_dir = snapshot_download(
            model_id,
            cache_dir=os.path.dirname(local_dir),
            revision='master'
        )

        # snapshot_download 通常返回实际存放文件的目录
        print(f"[OK] 下载完成: {result_dir}")

        return result_dir

    except Exception as e:
        print(f"\n[ERROR] 下载失败: {e}")
        print("提示: 检查网络或稍后重试")
        raise


def main():
    parser = argparse.ArgumentParser(description='下载预训练模型 (ModelScope)')
    parser.add_argument('--force', action='store_true', help='强制重新下载')
    parser.add_argument('--only-qwen', action='store_true', help='只下载 Qwen3-VL')
    parser.add_argument('--only-clip', action='store_true', help='只下载 CLIP')
    parser.add_argument('--output-dir', type=str, default='models', help='保存根目录')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VLA Agent 模型下载工具 (ModelScope)")
    print("="*60)
    
    if not check_modelscope():
        print("[ERROR] ModelScope 安装失败")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    try:
        # 1. 下载 Qwen3-VL 8B
        if not args.only_clip:
            print("\n>>> 步骤 1/2: Qwen3-VL-8B")
            results['qwen'] = download_model(
                model_id='Qwen/Qwen3-VL-8B-Instruct',
                local_dir=str(output_dir / 'Qwen3-VL-8B'),
                force=args.force
            )
        
        # 2. 下载 CLIP ViT-B/32
        if not args.only_qwen:
            print("\n>>> 步骤 2/2: CLIP ViT-B/32")
            results['clip'] = download_model(
                model_id='openai-mirror/clip-vit-base-patch32',
                local_dir=str(output_dir / 'clip-vit-base-patch32'),
                force=args.force
            )
            
        # 最后再清理一次，确保没有残留的临时文件
        clean_modelscope_temp_files(str(output_dir))  
        
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
