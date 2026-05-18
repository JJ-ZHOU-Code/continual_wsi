# Environment and Storage Notes

## Compute Server

- SSH host: A6000-tailscale
- Code entry path: /home/zjj/code/continual_wsi
- Actual code storage: /data_2_4T/data_zjj/continual_wsi/code/continual_wsi
- Main TCGA data: /data_1_16T/data_tcga

/home is full, so the code path under ~/code is a symlink to the data disk.

## Storage Policy

Use these paths consistently:

- Code: /home/zjj/code/continual_wsi
- Main outputs and intermediate results: /data_2_4T/data_zjj/continual_wsi
- Spillover outputs when /data_1_16T is tight: /data_2_4T/data_zjj/continual_wsi if available/created

Current space snapshot:

- /data_1_16T: about 735G free at last check
- /data_2_4T: about 1.6T free at last check

Avoid writing large caches to /home.

## Python Environments

Existing conda environments:

- clam: /home/zjj/miniconda3/envs/clam, Python 3.8, PyTorch 2.4.1+cu121
- loki: /home/zjj/miniconda3/envs/loki, Python 3.9, PyTorch 2.3.1+cu121

Default for current scripts:

`ash
/home/zjj/miniconda3/envs/clam/bin/python scripts/train_multicancer_smoke.py
`

## Dependency Management

Preferred order:

1. Reuse clam or loki if possible.
2. Use conda for CUDA/PyTorch-level packages.
3. Use uv for lightweight Python packages if installed or if we create a local tool environment.
4. Use mirror sources for package downloads when adding dependencies.

Suggested pip mirror:

`ash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
`

Suggested conda channels if needed:

`ash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
`

Do not modify global/base environments unless necessary. Prefer project-specific envs if new dependencies become substantial.

## Known Issue


vidia-smi sees A6000 GPUs, but the smoke test ran on CPU because PyTorch reported CUDA initialization failure in that SSH session. Before heavy training, debug CUDA visibility and driver/runtime access in the selected conda environment.
