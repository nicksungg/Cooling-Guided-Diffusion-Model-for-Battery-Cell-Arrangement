# [Cooling-Guided Diffusion Model for Battery Cell Arrangement](https://github.com/nicksungg/Cooling-Guided-Diffusion-Model-for-Battery-Cell-Arrangement)

> **Notice:** This repository contains the implementation of a Cooling-Guided Diffusion Model specifically designed to optimize battery cell layouts for enhanced cooling efficiency within Battery Thermal Management Systems (BTMS).

We introduce a novel approach using a **Cooling-Guided Diffusion Model** to optimize battery cell arrangements, a crucial step in improving the cooling efficiency and safety of battery packs. The model is based on a Denoising Diffusion Probabilistic Model (DDPM) guided by a classifier and a surrogate model to generate feasible and thermally efficient cell layouts.

This repository includes the full implementation of the model, which has been shown to outperform existing methods such as Tabular Denoising Diffusion Probabilistic Model (TabDDPM) and Conditional Tabular GAN (CTGAN) in key metrics like feasibility, diversity, and cooling efficiency. The dataset can be found [**here**](https://mitprod-my.sharepoint.com/:u:/g/personal/nicksung_mit_edu/EYcTBegU3cBNoA4TU-2c0qYBjWFuL6o0eaJM_qtS9DOPtA?e=U7GKWp).

For more detailed information, please refer to the following:

- **Sung, Nicholas, Zheng Liu, Pingfeng Wang, and Faez Ahmed.** "[Cooling-Guided Diffusion Model for Battery Cell Arrangement](https://arxiv.org/abs/2403.10566)." In *Proceedings of the ASME 2024 International Design Engineering Technical Conferences*, Washington, DC, USA, 2024.

## Citation

```bibtex
@inproceedings{Sung2024CoolingGuided,
  title={Cooling-Guided Diffusion Model for Battery Cell Arrangement},
  author={Nicholas Sung, Zheng Liu, Pingfeng Wang, Faez Ahmed},
  booktitle={Proceedings of the ASME 2024 International Design Engineering Technical Conferences},
  year={2024},
  address={Washington, DC, USA}
}
