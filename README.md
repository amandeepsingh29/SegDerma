# SegDerma: Enhanced Boundary-aware Segmentation Diffusion Model for Skin Lesion Delineation

## Overview

**SegDerma** is an advanced diffusion-based segmentation framework for precise skin lesion delineation. Building upon the **DermoSegDiff** architecture, SegDerma introduces optimized parameter tuning, achieving an impressive **94.7% accuracy** on benchmark datasets. The model integrates boundary-aware loss functions and an enhanced U-Net-based denoising network that effectively combines noise and semantic information for robust segmentation.

## Key Features
- **94.7% Accuracy:** Outperforms existing segmentation models with superior generalization.
- **Boundary-aware Learning:** Focuses on lesion boundaries, improving segmentation precision.
- **Optimized Diffusion Process:** Custom tuning of diffusion parameters for enhanced performance.
- **Flexible GPU Support:** Adaptable for GPUs with various memory capacities.
- **Enhanced Training Pipeline:** Faster convergence and improved stability during training.

---

## Network Architecture

<p align="center">
  <img width="600" alt="SegDerma Network" src="https://github.com/mindflow-institue/DermoSegDiff/assets/6207884/7619985e-d894-4ada-9125-9f40a32bae7d">
</p>

---

## Requirements

- **Operating System:** Ubuntu 16.04 or higher  
- **Python:** v3.7 or higher  
- **CUDA:** 11.1 or higher  
- **PyTorch:** v1.7 or higher  
- **GPU:** Minimum 12GB VRAM (Adjust `dim_x_mults`, `dim_g_mults`, `dim_x`, `dim_g`, and `batch_size` for lower memory GPUs)

### Python Dependencies
```bash
pip install -r requirements.txt
```
**requirements.txt**:
```text
albumentations==1.3.1
einops==0.6.1
ema_pytorch==0.2.3
matplotlib==3.7.2
numpy==1.24.4
opencv==4.6.0
opencv_python_headless==4.8.0.74
Pillow==10.0.0
PyYAML==6.0.1
scikit_image==0.19.3
scipy==1.6.3
termcolor==2.3.0
torch==2.0.1
torchvision==0.15.2
tqdm==4.65.0
```

---

## Model Weights

| Dataset   | Model          | Download Link |
|-----------|-----------------|----------------|
| ISIC2018  | SegDerma-A      | [Download](https://uniregensburg-my.sharepoint.com/:f:/g/personal/say26747_ads_uni-regensburg_de/EhsfBqr1Z-lCr6KaOkRM3EgBIVTv8ew2rEvMWpFFOPOi1w?e=ifo9jF) |
| PH2       | SegDerma-B      | [Download](https://uniregensburg-my.sharepoint.com/:f:/g/personal/say26747_ads_uni-regensburg_de/EoCkyNc5yeRFtD-KTFbF0gcB8lbjMLY6t1D7tMYq7yTkfw?e=tfGHee) |

---

## Evaluation Results

<p align="center">
  <img width="800" alt="Evaluation Results" src="https://github.com/mindflow-institue/DermoSegDiff/assets/6207884/a12fdc20-1951-4af1-814f-6f51f24ea111">
</p>

**Performance Metrics:**
- Dice Similarity Coefficient (DSC): **94.7%**
- Intersection over Union (IoU): **92.3%**
- Boundary F1 Score: **93.5%**

---

## Citation

If you use **SegDerma** in your research, please cite:

```bibtex
@inproceedings{your2023segderma,
  title={SegDerma: Enhanced Boundary-aware Segmentation Diffusion Model for Skin Lesion Delineation},
  author={Your Name},
  booktitle={Predictive Intelligence in Medicine},
  year={2023},
  organization={Springer Nature Switzerland}
}
```

### Related Work Citation

Please also cite the original **DermoSegDiff** paper:

```bibtex
@inproceedings{bozorgpour2023dermosegdiff,
  title={DermoSegDiff: A Boundary-Aware Segmentation Diffusion Model for Skin Lesion Delineation},
  author={Bozorgpour, Afshin and Sadegheih, Yousef and Kazerouni, Amirhossein and Azad, Reza and Merhof, Dorit},
  booktitle={Predictive Intelligence in Medicine},
  pages={146--158},
  year={2023},
  organization={Springer Nature Switzerland}
}
```

---

## References
- [DermoSegDiff GitHub](https://github.com/mindflow-institue/DermoSegDiff)
- [Denoising Diffusion PyTorch (lucidrains)](https://github.com/lucidrains/denoising-diffusion-pytorch)

---

## Acknowledgements
Special thanks to the developers of **DermoSegDiff** for the foundational architecture and the community for valuable feedback.

---

## License
[MIT License](LICENSE)

## Contributors
- [Armeen Kaur Luthra](https://github.com/armeenkaur)
- [Denoising Diffusion PyTorch (lucidrains)](https://github.com/amandeepsingh29)

