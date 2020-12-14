# Weakly-supervised learning for medical image segmentation (WSL4MIS).
* This project is developing, if you used this code in you research, please consider to cite the followings:

		@misc{wsl4mis2020,
		  title={{WSL4MIS}},
		  author={Luo, Xiangde},
		  howpublished={\url{https://github.com/Luoxd1996/WSL4MIS}},
		  year={2020}
		}
* More details of code and data will be provided later, thanks for your attention.
* Results (the result based on the ACDC dataset, 80 patients for training and 20 patients for validation (fold1))
![](https://github.com/Luoxd1996/WSL4MIS/blob/main/imgs/fold1_curve.png) 

**Note**: pCE means partially cross-entropy, TV means total variation, label denotes supervised by mask, scribble represents just supervised by scribbles.
# Dataset
* The ACDC dataset with mask annotation can be downloaded from: [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)
* The Scribble of ACDC can be downloaded from: [Scribble](https://gvalvano.github.io/wss-multiscale-adversarial-attention-gates/data)
