# MSDM-PAT

Train:


FS-DM

CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/ve/SIAT_kdata_ncsnpp.py --workdir=exp1 --mode=train --eval_folder=result


LS-DM

CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/ve/SIAT_kdata_ncsnpp.py --workdir=exp2 --mode=train --eval_folder=result
File path ： root= "./lzdata/train"




--mode=train
--eval_folder=result



Test:
python pc_sampling_pa.py

test默认调用exp1，exp2下的模型

weight下载：


链接：[https://pan.baidu.com/s/1b5UZB7EiQwQaia8fLwHM9g?pwd=1234](https://pan.baidu.com/s/1-9xK0AZToV0cB-Ei_QB3gg?pwd=1234) 
提取码：1234


Folder where the results are saved:result/2



### python 3.8:

torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2

tensorflow==2.8.0

ninja==1.10

ml-collections==0.1.1



### matlab2020b

k-Wave toolbox



文件夹python 、文件夹k-wave 和文件夹matlab中的所有文件要放在最外面的文件路径可以成功运行

