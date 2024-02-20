# Implamentation Code of "Discriminatively Matched Part Tokens for Pointly Supervised Instance Segmentation"

# Install
### Apex:
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
### Ops
We use some cude implementation for efficient calculation.
```
cd Connected_components_PyTorch
python setup.py install
```
### mmdet
Please follow [imTED](https://github.com/LiewFeng/imTED) to build the detection and segmentation environment.

