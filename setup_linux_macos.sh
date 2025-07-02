#!/bin/bash

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Python Project Auto-Setup (Mac/Linux)${NC}"
echo -e "${GREEN}=====================================${NC}"
echo

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo -e "${BLUE}Detected OS: $OS${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}Python not found. Please install Python first.${NC}"
    echo
    if [[ "$OS" == "macos" ]]; then
        echo "Mac installation options:"
        echo "  Option 1: brew install python"
        echo "  Option 2: Download from https://www.python.org/downloads/"
        echo "  Option 3: Install Miniconda: brew install --cask miniconda"
    elif [[ "$OS" == "linux" ]]; then
        echo "Linux installation options:"
        echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip"
        echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
        echo "  Arch: sudo pacman -S python python-pip"
        echo "  Or install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    fi
    echo
    exit 1
fi

echo -e "${GREEN}Python checked${NC}"

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
    echo -e "${RED}Conda not found on your machine${NC}"
    echo "Please install Miniconda or Anaconda:"
    if [[ "$OS" == "macos" ]]; then
        echo "  Mac: brew install --cask miniconda"
        echo "  Or download from: https://docs.conda.io/en/latest/miniconda.html"
    elif [[ "$OS" == "linux" ]]; then
        echo "  Linux: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "  Then: bash Miniconda3-latest-Linux-x86_64.sh"
    fi
    exit 1
fi

echo -e "${GREEN}Found Conda at: $CONDA_PATH${NC}"

# Add conda to PATH for this session
export PATH="$CONDA_PATH/bin:$PATH"

# Initialize conda for bash
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Check if main.py exists
if [[ ! -f "main.py" ]]; then
    echo -e "${RED}Error: main.py not found in current directory!${NC}"
    echo "Please make sure main.py is in the same folder as this script."
    exit 1
fi

# Check for dependency files
DEPS_FILE=""
DEPS_TYPE=""

if [[ -f "environment.yml" ]]; then
    DEPS_FILE="environment.yml"
    DEPS_TYPE="conda"
    echo -e "${GREEN}Found environment.yml - using Conda${NC}"
elif [[ -f "requirements.txt" ]]; then
    DEPS_FILE="requirements.txt"
    DEPS_TYPE="pip"
    echo -e "${YELLOW}Found requirements.txt - using pip${NC}"
else
    echo -e "${RED}Warning: No dependency file found!${NC}"
    echo "Please create either environment.yml or requirements.txt"
    exit 1
fi

# Set environment name
ENV_NAME="discord_bot"

echo
echo -e "${BLUE}Step 1: Setting up virtual environment...${NC}"
echo

# Check if conda environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}Conda environment '$ENV_NAME' already exists.${NC}"
    read -p "Do you want to update it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Updating environment..."
        conda env update -n "$ENV_NAME" -f "$DEPS_FILE"
    fi
else
    echo -e "${YELLOW}Creating new conda environment '$ENV_NAME'... Be patient, depending on your machine this can take a while.${NC}"
    if [[ "$DEPS_TYPE" == "conda" ]]; then
        echo -e "Creating env with file: $DEPS_TYPE"
        conda env create -f "$DEPS_FILE"
    else
        conda create -n "$ENV_NAME" python -y
        conda activate "$ENV_NAME"
        pip install -r "$DEPS_FILE"
    fi
    
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Failed to create conda environment!${NC}"
        exit 1
    fi
fi

echo
echo -e "${BLUE}Step 2: Setting up environment variables...${NC}"
echo

# Create .env file if it doesn't exist
if [[ ! -f ".env" ]]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat << 'EOF' > .env
# API Keys - Replace with your actual API keys
DISCORD_TOKEN="YOUR_DISCORD_TOKEN_HERE"
CLAUDE_API_KEY="YOUR_CLAUDE_API_KEY_HERE"
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
CUSTOM_API_KEY="YOUR_CUSTOM_API_KEY_HERE"
EOF
    echo
    echo -e "${GREEN}.env file created! Please edit .env and add your actual API keys.${NC}"
    echo -e "${RED}WARNING: Never commit .env files to version control!${NC}"
    echo
    
    # Try to open .env file with available editors
    if command -v code &> /dev/null; then
        echo "Opening .env in VS Code..."
        code .env
    else
        echo "Please edit .env manually with your preferred text editor"
    fi
else
    echo -e "${GREEN}.env file already exists.${NC}"
fi

# Create .gitignore if it doesn't exist
if [[ ! -f ".gitignore" ]]; then
    echo -e "${YELLOW}Creating .gitignore to protect sensitive files...${NC}"
    cat << 'EOF' > .gitignore
# Environment variables
.env

# Bot Data
bot_data/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
    echo -e "${GREEN}.gitignore created to protect your .env file!${NC}"
fi

echo
echo -e "${GREEN}Step 3: Installation complete!${NC}"
echo
echo -e "${YELLOW}Before running the application:${NC}"
echo "1. Edit the .env file and add your actual API keys"
echo "2. Run the launcher: ./launcher.sh"
echo
echo -e "${BLUE}To manually run the project:${NC}"
echo "  conda activate $ENV_NAME"
echo "  python main.py"
echo
echo -e "${GREEN}Setup completed successfully!${NC}"