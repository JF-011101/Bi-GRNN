# 基于图神经网络的多元缺失时间序列补全算法

## env
[conda 基本命令](https://www.likecs.com/show-308354496.html)


云服务器配置：
GPU 
2080 Ti-11G 数量： 1 显存： 11 GB
CPU
AMD EPYC 7601 实例内存： 31G
核心： 8 核
CUDA 11.6    python3.8
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
简单验证显卡是否可用：
```
python -c "import torch;print(torch.version.cuda)"
python -c "import torch;print(torch.cuda.is_available())"
python -c "import tensorflow as tf;print(tf.config.list_physical_devices('GPU'))"
```
请保持以下依赖库版本一致：
```
pip install pytorch-lightning==1.4
pip install torch==1.8
pip install fancyimpute==0.6
pip install torchmetrics==0.5
pip install pandas==1.4.2
pip install sklearn==0.0
```
## `run_baselines.py`
### mean 

```
python ./scripts/run_baselines.py --dataset air36 air bay bay_noise la la_noise --imputers mean > log_baseline_mean.log
python ./scripts/run_baselines.py --dataset air36 air bay bay_noise la la_noise --imputers mean --in-sample False > log_baseline_mean_F.log 
```
### knn
```
python ./scripts/run_baselines.py --dataset air36 air bay bay_noise la la_noise --imputers knn > log_baseline_knn.log
python ./scripts/run_baselines.py --dataset air36 air bay bay_noise la la_noise --imputers knn --in-sample False > log_baseline_knn_F.log
```
### mf
```
python ./scripts/run_baselines.py --dataset air36 air bay bay_noise la la_noise --imputers mf > log_baseline_mf.log
nohup python ./scripts/run_baselines.py --dataset bay_noise la la_noise --imputers mf > log_baseline_mf_add.log &
python ./scripts/run_baselines.py --dataset air36 --imputers mf --in-sample False > log_baseline_mf_F.log
```
### mice
```
nohup python ./scripts/run_baselines.py --dataset air36 --imputers mice > log_baseline_mice.log &
nohup python ./scripts/run_baselines.py --dataset air bay --imputers mice > log_baseline_mice.log &
nohup python ./scripts/run_baselines.py --dataset  bay_noise la la_noise --imputers mice > log_baseline_mice.log &
nohup python ./scripts/run_baselines.py --dataset air36 bay bay_noise la la_noise --imputers mice --in-sample False > log_baseline_mice_F.log &
nohup python ./scripts/run_baselines.py --dataset air --imputers mice --in-sample False > log_baseline_mice_F.log &
```


## `run_imputation.py`


### air
```
nohup python ./scripts/run_imputation.py --config config/bigrnn/air.yaml --in-sample False > log_bigrnn_airF.log &
nohup python ./scripts/run_imputation.py --config config/bigrnn/air.yaml --in-sample True > log_bigrnn_airT.log &
nohup python ./scripts/run_imputation.py --config config/brits/air.yaml --in-sample False > log_brits_airF.log &
nohup python ./scripts/run_imputation.py --config config/brits/air.yaml --in-sample True > log_brits_airT.log &
```
### air36
```
nohup python ./scripts/run_imputation.py --config config/bigrnn/air36.yaml --in-sample False > log_bigrnn_air36F.log &
nohup python ./scripts/run_imputation.py --config config/bigrnn/air36.yaml --in-sample True > log_bigrnn_air36T.log &	
nohup python ./scripts/run_imputation.py --config config/brits/air36.yaml --in-sample False > log_brits_air36F.log &
nohup python ./scripts/run_imputation.py --config config/brits/air36.yaml --in-sample True > log_brits_air36T.log &
```	
### la
``` 
nohup python ./scripts/run_imputation.py --config config/bigrnn/la_block.yaml > log_bigrnn_la_blockF.log &
nohup python ./scripts/run_imputation.py --config config/bigrnn/la_block.yaml > log_bigrnn_la_blockT.log &
nohup python ./scripts/run_imputation.py --config config/brits/la_block.yaml > log_brits_la_blockF.log &
nohup python ./scripts/run_imputation.py --config config/brits/la_block.yaml > log_brits_la_blockT.log &	
```	
### bay
```
nohup python ./scripts/run_imputation.py --config config/bigrnn/bay_block.yaml > log_bigrnn_bay_blockF.log &
nohup python ./scripts/run_imputation.py --config config/bigrnn/bay_block.yaml > log_bigrnn_bay_blockT.log &
nohup python ./scripts/run_imputation.py --config config/brits/bay_block.yaml > log_brits_bay_blockF.log &
nohup python ./scripts/run_imputation.py --config config/brits/bay_block.yaml > log_brits_bay_blockT.log &
```