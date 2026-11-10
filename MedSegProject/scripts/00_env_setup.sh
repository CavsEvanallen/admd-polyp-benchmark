#!/usr/bin/env bash
# 环境搭建（Ubuntu 22.04 + CUDA 11.8 镜像；AutoDL / 阿里云 / 腾讯云通用）
# 用法：bash 00_env_setup.sh 2>&1 | tee env_setup.log
set -euo pipefail

WORK=${WORK:-./work}      # 数据盘路径，按云平台改
mkdir -p "$WORK"/{data,ckpt,results,manifests,prompts,logs,third_party}
cd "$WORK"

# 1. conda 环境（镜像自带 miniconda 时直接用；没有就先装）
if ! command -v conda >/dev/null; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"; source "$HOME/miniconda3/bin/activate"; conda init bash
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n admd python=3.10
conda activate admd

# 2. PyTorch（与论文一致用 CUDA 11.8；mamba-ssm 对版本敏感，不要随意升级）
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# 3. Mamba 相关（编译需要 nvcc；若镜像没有 nvcc，改装预编译 wheel，见手册"常见坑"）
pip install packaging ninja
pip install causal-conv1d==1.1.1 --no-build-isolation
pip install mamba-ssm==1.1.1 --no-build-isolation

# 4. 通用依赖
pip install numpy==1.26.4 scipy pandas pillow opencv-python-headless scikit-image scikit-learn \
            matplotlib seaborn tqdm pyyaml timm==0.9.12 einops fvcore thop tensorboard

# 5. SAM 系基线（统一放 third_party；各仓库权重链接见 01_datasets.md 末尾）
cd third_party
git clone https://github.com/facebookresearch/segment-anything.git && pip install -e segment-anything
git clone https://github.com/ChaoningZhang/MobileSAM.git            && pip install -e MobileSAM
git clone https://github.com/chongzhou96/EdgeSAM.git                 && pip install -e EdgeSAM
git clone https://github.com/THU-MIG/RepViT.git                      # RepViT-SAM 在 RepViT/sam 子目录
git clone https://github.com/yformer/EfficientSAM.git
git clone https://github.com/bowang-lab/MedSAM.git                   && pip install -e MedSAM
git clone https://github.com/OpenGVLab/SAM-Med2D.git
# 域泛化 / 息肉基线
git clone https://github.com/Inha-CVAI/MADGNet.git
git clone https://github.com/DengPingFan/PraNet.git
git clone https://github.com/DengPingFan/Polyp-PVT.git
# 解码侧域泛化 SAM（审稿人会要求的两个近邻方法）
git clone https://github.com/yifangao112/DeSAM.git
git clone https://github.com/wkklavis/DAPSAM.git
cd ..

# 6. 组内 MMSeg 代码：从项目组拿到后放到 $WORK/mmseg_admd，并按 10_mmseg_interface.md 补开关
echo "环境装完。下一步：bash 检查 -> python -c 'import torch,mamba_ssm;print(torch.cuda.is_available())'"
