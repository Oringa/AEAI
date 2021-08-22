<!-- The **Autoencoder Adversarial Interpolation** (AEAI) is a novel approach for the generation of admissible interpolation in manifold data and was published in [ICML2021](https://proceedings.mlr.press/v139/oring21a).

This paper was written by [Alon Oring](https://www.linkedin.com/in/oringa/) under the supervision of [Prof. Zohar Yakhini](https://zohary.cswp.cs.technion.ac.il/) and [Prof. Yacov Hel-Or](https://faculty.idc.ac.il/toky/) from the Interdisciplinary Center Herzliya. -->

## Abstract
 
Autoencoders represent an effective approach for computing the underlying factors characterizing datasets of different types. The latent representation of autoencoders have been studied in the context of enabling interpolation between data points by decoding convex combinations of latent vectors. This interpolation, however, often leads to artifacts or produces unrealistic results during reconstruction. 

We argue that these incongruities are due to the structure of the latent space and because such naively interpolated latent vectors deviate from the data manifold. 

In this work, we propose a regularization technique that shapes the latent representation to follow a manifold that is consistent with the training images and that drives the manifold to be smooth and locally convex. This regularization not only enables faithful interpolation between data points, as we show herein, but can also be used as a general regularization technique to avoid overfitting or to produce new samples for data augmentation.

## Motivation

1. Autoencoder latent spaces are non-convex.
2. GANs are not bidirectional: To interpolate between two real data points, we must map the datapoints back into latent space where admissible interpolation can be performed. Such inverse mapping is not a part of the GAN framework. Additionally, the latent space of the GAN does not necessarily encode a smooth parameterization of the data. 

## Manifold Data Interpolation

Before presenting the proposed approach we would like to define what constitutes a proper interpolation between two data points. There are many possible paths between two points on the manifold. Even if we require the interpolations to be on a geodesic path, there might be infinitely many such paths between two points. Therefore, we relax the geodesic requirement and define less restrictive conditions.

Formally, assume we are given a dataset sampled from a target domain \\(\cal{X}\\). We are interested in interpolating between two data points \\( \boldsymbol x_i \\) and \\( \boldsymbol x_j \\) from \\(\cal{X}\\). Let the interpolated points be \\( \hat{\boldsymbol{x}}_{i \rightarrow j}(\alpha) \\) for \\( \alpha \in [0,1] \\) and let \\(P(\boldsymbol x)\\) be the probability that a data point \\(\boldsymbol x\\) belongs to \\(\cal{X}\\). We define an interpolation to be an **admissible interpolation** if \\( \hat{\boldsymbol{x}}_{i \rightarrow j}(\alpha) \\)






 <!-- \\( \hat{\boldsymbol{x}}_{i \rightarrow j}(\alpha)\\) for \\(\alpha \in [0,1]\\) and let \\(P(\boldsymbol x)\\) be the probability that a data point \\(\boldsymbol x\\) belongs to \\(\cal{X}\\). We define an interpolation to be an **admissible interpolation** if \\(\hat \boldsymbol x_{i \rightarrow j}(\alpha)\\) satisfies the following conditions: -->

<!-- 1. **Boundary conditions**: \\( \hat \boldsymbol x_{i \rightarrow j}(0)=\boldsymbol x_i \quad \text{and} \quad  \hat \boldsymbol x_{i \rightarrow j}(1)= \boldsymbol x_j \\) -->

\\( \hat x_i \\)
\\( \hat{\boldsymbol{x}}_i \\)



## Animations

| **AEAI** | **AAE** | **ACAI** |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/aeai_chess.gif"> | <img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/aae_chess.gif"> | <img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/acai_chess.gif"> 
| **VAE** | **GAIA** | **AMR** |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/beta_chess.gif"> | <img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/gaia_chess.gif"> | <img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/amr_chess.gif"> |

<hr style="border:1px solid gray">

| **AEAI** | **AAE** | **ACAI** |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/aeai_wood.gif"> | <img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/aae_wood.gif"> | <img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/acai_wood.gif"> 
| **VAE** | **GAIA** | **AMR** |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/beta_wood.gif"> | <img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/gaia_wood.gif"> | <img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/amr_wood.gif"> |

<hr style="border:1px solid gray">

| **AEAI** | **AAE** | **ACAI** |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/aeai_pole.gif"> | <img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/aae_pole.gif"> | <img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/acai_pole.gif"> 
| **VAE** | **GAIA** | **AMR** |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/beta_pole.gif"> | <img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/gaia_pole.gif"> | <img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/amr_pole.gif"> |



## BibTeX

If you find our work useful, please cite our paper:

```
@InProceedings{pmlr-v139-oring21a,
  title = 	 {Autoencoder Image Interpolation by Shaping the Latent Space},
  author =       {Oring, Alon and Yakhini, Zohar and Hel-Or, Yacov},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {8281--8290},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/oring21a/oring21a.pdf},
  url = 	 {https://proceedings.mlr.press/v139/oring21a.html}
  }
```