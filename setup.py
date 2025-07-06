#!/usr/bin/env python3
"""
Setup script for local development environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0

def setup_python_environment():
    """Set up Python virtual environment."""
    print("Setting up Python virtual environment...")
    
    if not run_command("python3 -m venv venv", check=False):
        print("Failed to create virtual environment")
        return False
    
    # Activate and install dependencies
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    install_cmd = f"{activate_cmd} && pip install --upgrade pip && pip install -r requirements.txt"
    
    if not run_command(install_cmd, check=False):
        print("Failed to install Python dependencies")
        return False
    
    print("✓ Python environment setup complete")
    return True

def setup_mongodb():
    """Set up MongoDB database."""
    print("Setting up MongoDB...")
    
    # Check if MongoDB is running
    if not run_command("mongosh --eval 'db.runCommand(\"ping\").ok' --quiet", check=False):
        print("MongoDB is not running. Please start MongoDB and try again.")
        print("On macOS: brew services start mongodb-community")
        print("On Ubuntu: sudo systemctl start mongod")
        return False
    
    print("✓ MongoDB setup complete")
    return True

def setup_redis():
    """Set up Redis for background jobs."""
    print("Setting up Redis...")
    
    # Check if Redis is running
    if not run_command("redis-cli ping", check=False):
        print("Redis is not running. Please start Redis and try again.")
        print("On macOS: brew services start redis")
        print("On Ubuntu: sudo systemctl start redis")
        return False
    
    print("✓ Redis setup complete")
    return True

def download_spacy_model():
    """Download spaCy language model."""
    print("Downloading spaCy language model...")
    
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    cmd = f"{activate_cmd} && python -m spacy download en_core_web_sm"
    
    if not run_command(cmd, check=False):
        print("Warning: Failed to download spaCy model. NLP features may not work.")
        return False
    
    print("✓ spaCy model download complete")
    return True

def create_env_file():
    """Create .env file from example."""
    if not Path(".env").exists() and Path(".env.example").exists():
        run_command("cp .env.example .env")
        print("✓ Created .env file from example")
        print("Please edit .env file with your specific configuration")

def main():
    """Main setup function."""
    print("🚀 Setting up Intelligent Data Processor")
    print("=" * 50)
    
    success = True
    
    # Create necessary directories
    Path("uploads").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)
    
    # Setup steps
    success &= setup_python_environment()
    success &= setup_mongodb()
    success &= setup_redis()
    success &= download_spacy_model()
    
    create_env_file()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file with your configuration")
        print("2. Run: source venv/bin/activate")
        print("3. Run: python main.py")
    else:
        print("\n❌ Setup encountered some issues. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
