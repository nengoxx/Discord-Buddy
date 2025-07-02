#!/bin/bash

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Discord Bot Launcher (Mac/Linux)${NC}"
echo "=================================="

# Function to find conda installation
find_conda() {
    local conda_paths=(
        "/opt/conda"              # Docker images (continuumio/miniconda3)
        "$HOME/anaconda3"
        "$HOME/miniconda3" 
        "$HOME/miniforge3"
        "/opt/anaconda3"
        "/opt/miniconda3"
        "/usr/local/anaconda3"
        "/usr/local/miniconda3"
        "/root/miniconda3"        # Sometimes in Docker as root
        "/conda"                  # Alternative Docker path
    )
    
    # Also check if conda is already in PATH
    if command -v conda &> /dev/null; then
        # Get conda's base path
        local conda_info=$(conda info --base 2>/dev/null)
        if [[ -n "$conda_info" && -d "$conda_info" ]]; then
            echo "$conda_info"
            return 0
        fi
    fi
    
    for path in "${conda_paths[@]}"; do
        if [[ -f "$path/bin/conda" ]]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

# Try to find conda installation
CONDA_PATH=$(find_conda)
if [[ $? -ne 0 ]]; then
    echo -e "${RED}Error: Conda not found on your machine${NC}"
    echo "Please install Miniconda or Anaconda:"
    echo "  Mac: brew install --cask miniconda"
    echo "  Linux: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

echo -e "${GREEN}Found Conda at: $CONDA_PATH${NC}"

# Add conda to PATH for this session
export PATH="$CONDA_PATH/bin:$PATH"

# Check if discord_bot environment exists
ENV_PATH="$CONDA_PATH/envs/discord_bot"
if [[ ! -d "$ENV_PATH" ]]; then
    echo -e "${RED}Error: discord_bot environment not found at $ENV_PATH${NC}"
    echo "Please create the environment first:"
    echo "  conda create -n discord_bot python"
    echo "Or run the setup script: ./setup.sh"
    exit 1
fi

# Activate discord_bot environment
echo -e "${YELLOW}Activating discord_bot environment...${NC}"
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate discord_bot

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Failed to activate conda environment${NC}"
    exit 1
fi

echo -e "${GREEN}Environment activated. Python path:${NC}"
which python

# Check if main.py exists
if [[ ! -f "main.py" ]]; then
    echo -e "${RED}Error: main.py not found in current directory${NC}"
    exit 1
fi

echo -e "${YELLOW}Running main.py...${NC}"
python main.py
