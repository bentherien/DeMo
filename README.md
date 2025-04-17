# Install

```
apt-get update
apt install tmux vim rsync htop -y
tmux

mkdir demo_install
cd demo_install

wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh
bash Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -b -p $PWD/miniconda3
source $PWD/miniconda3/bin/activate



git clone  https://github.com/bentherien/DeMo

# install OLMO
git clone https://github.com/allenai/OLMo
cd OLMo
git checkout 46f06cb
cd ../DeMo

python -m 0001-DeMo.patch

pip install torch einops

```



# DeMo
This package contains the supplementary material for [DeMo: Decoupled Momentum Optimization](https://arxiv.org/abs/2411.19870) (arXiv)

A standalone PyTorch optimizer is provided in `demo.py`.

To reproduce the experiments in the paper, apply `0001-DeMo.patch` to https://github.com/allenai/OLMo/commit/46f06cbc3b42ed94a2400dec4aa479197d1ba0b6.
To launch the training jobs run `torchrun --nodes=8 --nproc-per-node=8 scripts/train.py CONFIG_FILE` where `CONFIG_FILE` is any of the `.yaml` files provided in this package.

For implementation in other PyTorch training pipelines, the standalone DeMo optimizer can be used as-is, the only additional modification needed is to disable the native Distributed Data Parallel gradient synchronization/all-reduce.

Future updates will be on the [DisTrO](https://github.com/NousResearch/DisTrO) repo.
