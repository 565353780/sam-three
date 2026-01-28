cd ..
git clone https://github.com/ronghanghu/cc_torch.git

pip install ninja

pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install opencv-python einops psutil matplotlib \
  pandas scikit-image scikit-learn decord pycocotools \
  tqdm

cd cc_torch
python setup.py install

cd ../sam-three
pip install -e .
