#!/bin/bash
# Setup script for elongation-rebar project
# Installs system and Python dependencies

set -e

# Update package list
sudo apt-get update

# Install system dependencies
sudo apt-get install -y xdg-utils ffmpeg ttf-mscorefonts-installer feh

# Install Python dependencies
pip install -r requirements.txt

echo "\n✅ Setup complete! You can now run the analysis scripts." 