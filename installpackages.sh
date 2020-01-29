pip install transformers
pip install seqeval
pip install tensorboardx
pip install torchvision==0.4.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html --no-cache-dir
rm -rf .git
rm -rf apex
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
pip3 install simpletransformers
pip install afinn