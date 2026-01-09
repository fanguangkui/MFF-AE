# 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18193972.svg)](https://doi.org/10.5281/zenodo.18193972)

## Installation

Please install the anaconda firstly.
```shell
conda create -n ppOD python=3.7 
conda activate ppOD
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tensorflow
```


### optional

if using **log2_qnorm** or **qnorm** in **scale_strategy**, please install R and
refer this [link](https://bioconductor.org/packages/release/bioc/html/limma.html) to install limma.

##### windows
- Install [R](https://cloud.r-project.org/bin/windows/base/) software.

- A work around is to download [rpy2](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rpy2) from the Unofficial Windows Binaries for Python Extension Packages to the current working directory. Then use the following command to install rpy2 from the downloaded file:
    ```shell
    cd extension
    pip install rpy2-2.9.5-cp37-cp37m-win_amd64.whl
    ```
- please following [this blog](http://joonro.github.io/blog/posts/install-rpy2-windows-10/) to config the enviroment.

##### Linux
- Install R Software
  check R version that the system can get, we need it >=4.2: 
    ```shell
    sudo apt-get remove r-base r-base-core r-base-dev r-recommended
    
    apt policy r-base
    ```
- install R
    ```shell
    # update indices
    sudo apt update -qq
    # install two helper packages we need
    sudo apt install --no-install-recommends software-properties-common dirmngr
    # add the signing key (by Michael Rutter) for these repos
    # To verify key, run gpg --show-keys /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
    # Fingerprint: E298A3A825C0D65DFD57CBB651716619E084DAB9
    wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
    # add the R 4.0 repo from CRAN -- adjust 'focal' to 'groovy' or 'bionic' as needed
    sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
    sudo apt update
    sudo apt install --no-install-recommends r-base
    ```
- install limma by referring [this link](https://bioconductor.org/packages/limma/)

- install rpy2    

    ```shell
    pip install rpy2
    ```

# Data Availability

The datasets used in this paper are available for download from the link below:

- **Download Link:** - **Platform:** Baidu Netdisk - **Link:** [Download Here](https://pan.baidu.com/s/1gbL6QSOXDaI6NTphrMDyQw?pwd=n322 )  

**Instructions:** 

1. Download  the `datasets` folder. 
2.  Place the unzipped `datasets` folder in the project root directory. 
3. Verify the path: `./datasets/Fake/` and `./datasets/Format/`

# Run Outliers Detection

To reproduce the experimental results of MFF-AE and compare it with other models (Ablation studies & Case studies), please follow the steps below. The configuration files are located in `../../configs/`.
## 1.Performance Evaluation (Simulated Data)
``` bash
# Standard AutoEncoder Baseline
python3 train_OD.py -c ../../configs/Hela/Standard_AutoEncoder.yaml data_dir datasets/Fake train_status false infer_status false eval_status true infer_datafile fake-shuffle_ratio-0.01-repeat_time-1.gz

# Ablation: AE + Classification Loss
python3 train_OD.py -c ../../configs/Hela/ResAE/AE_CLS.yaml data_dir datasets/Fake train_status false infer_status false eval_status true infer_datafile fake-shuffle_ratio-0.01-repeat_time-1.gz

# Ablation: AE + Residual Loss(MFF)
python3 train_OD.py -c ../../configs/Hela/ResAE/AE_CLS_ResLoss.yaml data_dir datasets/Fake train_status false infer_status false eval_status true infer_datafile fake-shuffle_ratio-0.01-repeat_time-1.gz

# MFF-AE (Full Model)
python3 train_OD.py -c ../../configs/Hela/ResAE/AE_ResLoss.yaml data_dir datasets/Fake train_status false infer_status false eval_status true infer_datafile fake-shuffle_ratio-0.01-repeat_time-1.gz

```

## 
## 2. Detect outlier in LADC Data 

Apply the model to detect outliers in the Lung Adenocarcinoma (LADC) dataset.

``` bash
# Detect in LADC Normal samples (N)
python3 train_OD.py -c ../../configs/LADC/AE/AE_CLS_ResLoss_N.yaml data_dir datasets/Format train_status true infer_status false eval_status false

# Detect in LADC Tumor samples (T)
python3 train_OD.py -c ../../configs/LADC/AE/AE_CLS_ResLoss_T.yaml data_dir datasets/Format train_status true infer_status false eval_status false
```



