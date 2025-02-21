
# MixMamba-Fewshot: Mamba and Attention Mixer-based Method with Few-Shot Learning for Bearing Fault Diagnosis

This is our implemented source code for the paper "[MixMamba-Fewshot: Mamba and Attention Mixer-based Method with Few-Shot Learning for Bearing Fault Diagnosis](https://doi.org/10.1016/j.compeleceng.2024.110004)" published in the Journal of Applied Intelligence
## Methodology
![plot](images/model_revise.png)


## Environment
```bash 
conda create -n MAMBA python=3.10.12 -y
conda activate MAMBA
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset
[CWRU Download Link](https://engineering.case.edu/bearingdatacenter)

## Getting Started
### Installation

``` bash
git clone https://github.com/linhthan216/Mamba-Bearing.git
cd Mamba-Bearing
```

### Data
- You can install data for your experiments via command
```
gdown 1VZ5GbFPZV1lfkkyHpGtIiTuql4vYoyX3
unzip CWRU.zip
```

### Training
```bash
chmod +x train.sh
```
- 1-shot training

```bash
bash train.sh 1 
```
- 5-shot training
```bash
bash train.sh 5
```

### Testing

```bash
chmod +x test.sh
```
- 1-shot testing
```bash
bash test.sh 1 
```
- 5-shot testing
```bash
bash test.sh 5
```

## Contact
Please feel free to contact me via email linh.tn212860@sis.hust.edu.vn if you need anything related to this repo!
## Citation
If you feel this code is useful, please give us 1 ⭐ and cite our paper.
```bash


```

