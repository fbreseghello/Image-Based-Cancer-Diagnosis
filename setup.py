"""
Setup script for the Image-Based Cancer Diagnosis project.
Ensures all dependencies are installed and directories are created.
"""

import subprocess
import sys
from pathlib import Path

def create_directories():
    """Create necessary project directories."""
    directories = [
        "models",
        "models/evaluation",
        "logs",
        "sample_images/benign",
        "sample_images/malignant",
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")
        else:
            print(f"○ Directory already exists: {dir_path}")


def install_dependencies():
    """Install required Python packages."""
    print("\nInstalling dependencies...")
    print("-" * 70)
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✓ All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing dependencies: {e}")
        sys.exit(1)


def verify_installation():
    """Verify that key packages are installed."""
    required_packages = [
        "tensorflow",
        "streamlit",
        "numpy",
        "pillow",
        "matplotlib",
        "sklearn",
    ]
    
    print("\nVerifying installation...")
    print("-" * 70)
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ Warning: Some packages are missing: {missing_packages}")
        return False
    else:
        print("\n✓ All required packages are installed!")
        return True


def main():
    """Main setup function."""
    print("=" * 70)
    print("Image-Based Cancer Diagnosis - Setup")
    print("=" * 70)
    
    print("\n[1/3] Creating directories...")
    create_directories()
    
    print("\n[2/3] Installing dependencies...")
    install_dependencies()
    
    print("\n[3/3] Verifying installation...")
    success = verify_installation()
    
    print("\n" + "=" * 70)
    if success:
        print("Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add your images to sample_images/benign and sample_images/malignant")
        print("2. Train the model: python train_model.py")
        print("3. Run the app: python run_app.py")
    else:
        print("Setup completed with warnings.")
        print("Please check the missing packages above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
