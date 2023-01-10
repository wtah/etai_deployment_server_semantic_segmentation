#!/bin/bash
pip install -U opencv-python
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
mkdir ext_repo
cd ext_repo
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# On macOS, you may need to prepend the above commands with a few environment variables:
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...

python -m pip install 'git+https://github.com/cocodataset/panopticapi.git'
python -m pip install 'git+https://github.com/mcordts/cityscapesScripts.git'

git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../..