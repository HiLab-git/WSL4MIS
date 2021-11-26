# Weakly-supervised learning for medical image segmentation (WSL4MIS).
* This project was originally developed for our previous works [[Paper](https://arxiv.org/pdf/2111.02403.pdf) & [Dataset](https://github.com/HiLab-git/WORD)]. If you use this codebase in your research, please cite the following works:
 
		@article{luo2021word,
		  title={WORD: Revisiting Organs Segmentation in the Whole Abdominal Region},
		  author={Luo, Xiangde and Liao, Wenjun and Xiao, Jianghong and Song, Tao and Zhang, Xiaofan and Li, Kang and Wang, Guotai and Zhang, Shaoting},
		  journal={arXiv preprint arXiv:2111.02403},
		  year={2021}
		}
		@misc{wsl4mis2020,
		  title={{WSL4MIS}},
		  author={Luo, Xiangde},
		  howpublished={\url{https://github.com/Luoxd1996/WSL4MIS}},
		  year={2020}
		}
		
# Dataset
* The ACDC dataset with mask annotations can be downloaded from: [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).
* The Scribble annotations of ACDC can be downloaded from: [Scribble](https://gvalvano.github.io/wss-multiscale-adversarial-attention-gates/data).
* The data processing code in [Here](https://github.com/Luoxd1996/WSL4MIS/blob/main/code/dataloaders/acdc_data_processing.py) or email [me](luoxd1996@gmail.com) for pre-processed data.
# Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python == 3.6 
* Efficientnet-Pytorch `pip install efficientnet_pytorch`
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Clone the repo:
```
git clone https://github.com/Luoxd1996/WSL4MIS
cd WSL4MIS
```
2. Download and pre-process data and put the data in  `../data/ACDC`.

3. Train the model
```
cd code
python train_XXX_2D.py
```

4. Test the model
```
python test_2D_fully.py
```
5. Training curves on the fold1
![](https://github.com/Luoxd1996/WSL4MIS/blob/main/imgs/fold1_curve.png) 
**Note**: pCE means partially cross-entropy, TV means total variation, label denotes supervised by mask, scribble represents just supervised by scribbles.

# Implemented methods
* **pCE**
* **pCE + TV**
* **pCE + Entropy Minimization**
* **pCE + GatedCRFLoss**

# Acknowledgement
* The GatedCRFLoss is adapted from [GatedCRFLoss](https://github.com/LEONOB2014/GatedCRFLoss) for medical image segmentation.
* The codebase is adapted from our previous work [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).
