# ModelArts Env
```
conda activate MindSpore
pip install mindspore-2.3.1-cp38-cp38-linux_aarch64.whl
```

贵阳一不能下载北京四的内容，所以需要手动下载 `https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.1/MindSpore/unified/aarch64/mindspore-2.3.1-cp38-cp38-linux_aarch64.whl` 然后上传到modelarts服务器上

# How to run
```
# 安装 mindnlp
git clone https://github.com/mindspore-lab/mindnlp.git
cd mindnlp
bash scripts/build_and_reinstall.sh

# 下载 dna_lm的脚本
git clone https://github.com/neoming/mindnlp_dna_lm.git
cd mindnlp_dna_lm
python main.py
```

# Tree file

+ README.md: 本文件
+ train.log: CPU 服务器Log
+ modelarts.log: NPU 服务器Log
+ dna_lm.ipynb: note book
+ main.py: note book中的代码一致的python脚本