# Weakly-supervised learning for medical image segmentation (WSL4MIS).
* This project was originally developed for our two previous works **[WORD](https://www.sciencedirect.com/science/article/pii/S1361841522002705)** (**MedIA2022**) and **[WSL4MIS](https://link.springer.com/chapter/10.1007/978-3-031-16431-6_50)** (**MICCAI2022**). If you use this project in your research, please cite the following works:

		@article{luo2022scribbleseg,
		title={Scribble-Supervised Medical Image Segmentation via Dual-Branch Network and Dynamically Mixed Pseudo Labels Supervision},
		author={Xiangde Luo, Minhao Hu, Wenjun Liao, Shuwei Zhai, Tao Song, Guotai Wang, Shaoting Zhang},
		journal={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
		year={2022},
		pages={528--538}}
		
		@article{luo2022word,
		title={{WORD}: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from CT image},
		author={Xiangde Luo, Wenjun Liao, Jianghong Xiao, Jieneng Chen, Tao Song, Xiaofan Zhang, Kang Li, Dimitris N. Metaxas, Guotai Wang, and Shaoting Zhang},
		journal={Medical Image Analysis},
		volume={82},
		pages={102642},
		year={2022},
		publisher={Elsevier}}
		
		@misc{wsl4mis2020,
		title={{WSL4MIS}},
		author={Luo, Xiangde},
		howpublished={\url{https://github.com/Luoxd1996/WSL4MIS}},
		year={2021}}
		
* A re-implementation of this work based on the [PyMIC](https://github.com/HiLab-git/PyMIC) can be found here ([WSLDMPLS](https://github.com/HiLab-git/PyMIC_examples/tree/main/seg_wsl/ACDC)).

# Dataset
* The ACDC dataset with mask annotations can be downloaded from: [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).
* The Scribble annotations of ACDC can be downloaded from: [Scribble](https://gvalvano.github.io/wss-multiscale-adversarial-attention-gates/data).
* The data processing code in [Here](https://github.com/Luoxd1996/WSL4MIS/blob/main/code/dataloaders/acdc_data_processing.py)  the pre-processed ACDC data in [Here](https://github.com/HiLab-git/WSL4MIS/tree/main/data/ACDC).
* **To simulate the scribble annotation for other datasets, we further provide the simulation code at [Here](https://github.com/HiLab-git/WSL4MIS/blob/main/code/scribbles_generator.py)**.
# Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python >= 3.6 
* Efficientnet-Pytorch `pip install efficientnet_pytorch`
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Clone this project.
```
git clone https://github.com/HiLab-git/WSL4MIS
cd WSL4MIS
```
2. Data pre-processing os used or the processed data.
```
cd code
python dataloaders/acdc_data_processing.py
```
3. Train the model
```
cd code
bash train_wss.sh # train model with scribble or dense annotations.
bash train_ssl.sh  # train model with mix-supervision (mask annotations and without annotation).
```

4. Test the model
```
python test_2D_fully.py --sup_type scribble/label --exp ACDC/the trained model fold --model unet
python test_2D_fully_sps.py --sup_type scribble --exp ACDC/the trained model fold --model unet_cct
```

5. Training curves on the fold1:
![](https://github.com/Luoxd1996/WSL4MIS/blob/main/imgs/fold1_curve.png) 
**Note**: pCE means partially cross-entropy, TV means total variation, label denotes supervised by mask, scribble represents just supervised by scribbles.

# Implemented methods
* [**pCE**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tang_Normalized_Cut_Loss_CVPR_2018_paper.pdf)
* [**pCE + TV**](https://arxiv.org/pdf/1605.01368.pdf)
* [**pCE + Entropy Minimization**](https://arxiv.org/pdf/2111.02403.pdf)
* [**pCE + GatedCRFLoss**](https://github.com/LEONOB2014/GatedCRFLoss)
* [**pCE + Intensity Variance Minimization**](https://arxiv.org/pdf/2111.02403.pdf)
* [**pCE + Random Walker**](http://vision.cse.psu.edu/people/chenpingY/paper/grady2006random.pdf)
* [**pCE + MumfordShah_Loss**](https://arxiv.org/pdf/1904.02872.pdf)
* [**Scribble2Label**](https://arxiv.org/pdf/2006.12890.pdf)
* [**USTM**](https://www.sciencedirect.com/science/article/pii/S0031320321005215)
* [**ScribbleVC**](https://github.com/HUANGLIZI/ScribbleVC)

# Acknowledgement
* The GatedCRFLoss is adapted from [GatedCRFLoss](https://github.com/LEONOB2014/GatedCRFLoss) for medical image segmentation.
* The codebase is adapted from our previous work [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).
* The WORD dataset will be presented at [WORD](https://github.com/HiLab-git/WORD).
