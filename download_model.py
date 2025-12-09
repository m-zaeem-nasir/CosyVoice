#!/usr/bin/env python3
"""
CosyVoice Model Downloader

This script downloads pretrained CosyVoice models from ModelScope.
Supports both SDK-based and git-based downloads.

Usage:
    python download_model.py --all                          # Download all models
    python download_model.py --model cosyvoice2             # Download CosyVoice2-0.5B
    python download_model.py --model cosyvoice-300m         # Download CosyVoice-300M
    python download_model.py --model sft                    # Download CosyVoice-300M-SFT
    python download_model.py --model instruct               # Download CosyVoice-300M-Instruct
    python download_model.py --model ttsfrd                 # Download CosyVoice-ttsfrd
    python download_model.py --method git                   # Use git clone instead of SDK
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


# Model configurations
MODELS = {
    'cosyvoice2': {
        'name': 'CosyVoice2-0.5B',
        'repo': 'iic/CosyVoice2-0.5B',
        'local_dir': 'pretrained_models/CosyVoice2-0.5B',
        'git_url': 'https://www.modelscope.cn/iic/CosyVoice2-0.5B.git',
        'description': 'Latest CosyVoice 2.0 model (Recommended)'
    },
    'cosyvoice-300m': {
        'name': 'CosyVoice-300M',
        'repo': 'iic/CosyVoice-300M',
        'local_dir': 'pretrained_models/CosyVoice-300M',
        'git_url': 'https://www.modelscope.cn/iic/CosyVoice-300M.git',
        'description': 'Base CosyVoice 300M model'
    },
    'sft': {
        'name': 'CosyVoice-300M-SFT',
        'repo': 'iic/CosyVoice-300M-SFT',
        'local_dir': 'pretrained_models/CosyVoice-300M-SFT',
        'git_url': 'https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git',
        'description': 'CosyVoice 300M SFT (Supervised Fine-Tuning) model'
    },
    'instruct': {
        'name': 'CosyVoice-300M-Instruct',
        'repo': 'iic/CosyVoice-300M-Instruct',
        'local_dir': 'pretrained_models/CosyVoice-300M-Instruct',
        'git_url': 'https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git',
        'description': 'CosyVoice 300M Instruct model'
    },
    'ttsfrd': {
        'name': 'CosyVoice-ttsfrd',
        'repo': 'iic/CosyVoice-ttsfrd',
        'local_dir': 'pretrained_models/CosyVoice-ttsfrd',
        'git_url': 'https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git',
        'description': 'Text normalization resources'
    }
}


def check_git_lfs():
    """Check if git-lfs is installed"""
    try:
        result = subprocess.run(['git', 'lfs', 'version'],
                              capture_output=True,
                              text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_via_sdk(model_key):
    """Download model using ModelScope SDK"""
    model = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Downloading {model['name']} via ModelScope SDK...")
    print(f"Description: {model['description']}")
    print(f"{'='*60}\n")

    try:
        from modelscope import snapshot_download

        # Create pretrained_models directory if not exists
        os.makedirs('pretrained_models', exist_ok=True)

        # Download model
        snapshot_download(model['repo'], local_dir=model['local_dir'])

        print(f"\n✓ Successfully downloaded {model['name']} to {model['local_dir']}")
        return True

    except ImportError:
        print("\n✗ Error: modelscope package not installed")
        print("Please install it with: pip install modelscope")
        return False
    except Exception as e:
        print(f"\n✗ Error downloading {model['name']}: {str(e)}")
        return False


def download_via_git(model_key):
    """Download model using git clone"""
    model = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Downloading {model['name']} via git clone...")
    print(f"Description: {model['description']}")
    print(f"{'='*60}\n")

    # Check if git-lfs is installed
    if not check_git_lfs():
        print("✗ Warning: git-lfs is not installed!")
        print("Please install git-lfs first:")
        print("  - Ubuntu/Debian: sudo apt-get install git-lfs")
        print("  - macOS: brew install git-lfs")
        print("  - Windows: download from https://git-lfs.github.com/")
        print("\nThen run: git lfs install")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            return False

    try:
        # Create pretrained_models directory if not exists
        os.makedirs('pretrained_models', exist_ok=True)

        # Clone repository
        cmd = ['git', 'clone', model['git_url'], model['local_dir']]
        result = subprocess.run(cmd, check=True)

        print(f"\n✓ Successfully downloaded {model['name']} to {model['local_dir']}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error cloning {model['name']}: {str(e)}")
        return False
    except FileNotFoundError:
        print("\n✗ Error: git is not installed")
        return False


def install_ttsfrd():
    """Install ttsfrd package after downloading"""
    ttsfrd_dir = Path('pretrained_models/CosyVoice-ttsfrd')

    if not ttsfrd_dir.exists():
        print("\n✗ CosyVoice-ttsfrd directory not found. Please download it first.")
        return False

    print(f"\n{'='*60}")
    print("Installing ttsfrd package...")
    print(f"{'='*60}\n")

    resource_zip = ttsfrd_dir / 'resource.zip'

    if not resource_zip.exists():
        print(f"✗ resource.zip not found in {ttsfrd_dir}")
        return False

    try:
        # Unzip resource
        print("Extracting resource.zip...")
        subprocess.run(['unzip', '-o', str(resource_zip), '-d', str(ttsfrd_dir)],
                      check=True)

        # Install dependencies
        dependency_whl = ttsfrd_dir / 'ttsfrd_dependency-0.1-py3-none-any.whl'
        ttsfrd_whl = ttsfrd_dir / 'ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl'

        if dependency_whl.exists():
            print(f"Installing {dependency_whl.name}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', str(dependency_whl)],
                          check=True)

        if ttsfrd_whl.exists():
            print(f"Installing {ttsfrd_whl.name}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', str(ttsfrd_whl)],
                          check=True)

        print("\n✓ Successfully installed ttsfrd package")
        print("Note: ttsfrd is optional. If not installed, WeText will be used by default.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing ttsfrd: {str(e)}")
        return False
    except FileNotFoundError:
        print("\n✗ Error: unzip command not found")
        print("Please install unzip first or manually extract resource.zip")
        return False


def list_models():
    """List all available models"""
    print("\nAvailable models:")
    print("="*60)
    for key, model in MODELS.items():
        status = "✓" if Path(model['local_dir']).exists() else "✗"
        print(f"{status} {key:20} - {model['description']}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Download CosyVoice pretrained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--model',
        choices=list(MODELS.keys()),
        help='Model to download'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all models'
    )

    parser.add_argument(
        '--method',
        choices=['sdk', 'git'],
        default='sdk',
        help='Download method (default: sdk)'
    )

    parser.add_argument(
        '--install-ttsfrd',
        action='store_true',
        help='Install ttsfrd package after downloading (Linux only)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available models and their download status'
    )

    args = parser.parse_args()

    # List models
    if args.list:
        list_models()
        return

    # Determine which models to download
    if args.all:
        models_to_download = list(MODELS.keys())
    elif args.model:
        models_to_download = [args.model]
    else:
        parser.print_help()
        print("\n")
        list_models()
        return

    # Choose download function
    download_func = download_via_git if args.method == 'git' else download_via_sdk

    # Download models
    success_count = 0
    for model_key in models_to_download:
        if download_func(model_key):
            success_count += 1

    # Install ttsfrd if requested
    if args.install_ttsfrd and 'ttsfrd' in models_to_download:
        install_ttsfrd()

    # Summary
    print(f"\n{'='*60}")
    print(f"Download Summary: {success_count}/{len(models_to_download)} models downloaded successfully")
    print(f"{'='*60}\n")

    if success_count > 0:
        print("Models are ready to use!")
        print("\nQuick start:")
        print("  python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B")
        print("\nOr check README.md for more usage examples.")


if __name__ == '__main__':
    main()
