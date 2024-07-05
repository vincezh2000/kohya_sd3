#!/usr/bin/bash
export PIP_INDEX_URL="https://mirror.baidu.com/pypi/simple"
export HF_ENDPOINT="https://hf-mirror.com"

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
create_venv=true

while [ -n "$1" ]; do
    case "$1" in
        --disable-venv)
            create_venv=false
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if $create_venv; then
    echo "Creating python venv..."
    python3 -m venv venv
    source "$script_dir/venv/bin/activate"
    echo "active venv"
fi

echo "Installing torch & xformers..."

cuda_version=$(nvcc --version | grep 'release' | sed -n -e 's/^.*release \([0-9]\+\.[0-9]\+\),.*$/\1/p')
cuda_major_version=$(echo "$cuda_version" | awk -F'.' '{print $1}')
cuda_minor_version=$(echo "$cuda_version" | awk -F'.' '{print $2}')

echo "Cuda Version:$cuda_version"

echo "install torch 2.2.2+cu121"
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.25.post1

echo "Installing deps..."
cd "$script_dir/sd-scripts" || exit

pip install --upgrade -r requirements.txt
pip install --upgrade lycoris-lora dadaptation fastapi uvicorn wandb
pip install --upgrade --no-deps pytorch-optimizer
pip install --upgrade schedulefree -i https://pypi.org/simple
pip install --upgrade git+https://github.com/huggingface/accelerate.git@main#egg=accelerate

cd "$script_dir" || exit

echo "Install completed"
