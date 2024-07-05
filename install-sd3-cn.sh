#!/usr/bin/bash
export HF_HOME="huggingface"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export PIP_INDEX_URL="https://pypi.mirrors.ustc.edu.cn/simple"
export HF_ENDPOINT="https://hf-mirror.com"

script_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
create_venv=true

function installFail {
    echo "安装失败。"
    read -r
    exit
}

function check {
    local errorInfo=$1
    if [ $? -ne 0 ]; then
        echo "$errorInfo"
        installFail
    fi
}

if $create_venv && [ ! -d "venv" ]; then
    echo "正在创建虚拟环境..."
    python3 -m venv venv
    check "创建虚拟环境失败，请检查 python 是否安装完整以及 python 版本是否为64位版本的python 3.10、或python的目录是否在环境变量PATH内。"
fi

source "$script_dir/venv/bin/activate"
check "激活虚拟环境失败。"

cd "$script_dir/sd-scripts" || exit
echo "安装程序所需依赖（已进行国内加速，若在国外或无法使用加速源，请使用 install.ps1 脚本版本）"

read -p "是否需要安装 Torch+xformers? 如果本次为首次安装请选择 y，如果本次为升级依赖安装则选择 n。[y/n] (默认为 y): " install_torch
install_torch=${install_torch:-y}
if [[ "$install_torch" =~ ^[yY]$ || -z "$install_torch" ]]; then
    pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html
    check "torch 安装失败，请删除 venv 文件夹后重新运行。"
    pip install --no-deps xformers==0.0.26.post1
    check "xformers 安装失败。"
fi

pip install --upgrade -r requirements.txt
check "其它依赖安装失败。"
pip install --upgrade dadaptation
check "Lion、dadaptation 优化器安装失败。"
pip install --upgrade --pre lycoris-lora -i https://pypi.org/simple
check "lycoris 安装失败。"
pip install --upgrade fastapi uvicorn scipy
check "UI 所需依赖安装失败。"
pip install --upgrade wandb
check "wandb 安装失败。"
pip install --upgrade --no-deps pytorch-optimizer
check "pytorch-optimizer 安装失败。"
pip install --upgrade schedulefree -i https://pypi.org/simple
check "schedulefree 安装失败。"

echo "安装完成"
read -r
