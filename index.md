<!-- The **Autoencoder Adversarial Interpolation** (AEAI) is a novel approach for the generation of admissible interpolation in manifold data and was published in [ICML2021](https://proceedings.mlr.press/v139/oring21a).

This paper was written by [Alon Oring](https://www.linkedin.com/in/oringa/) under the supervision of [Prof. Zohar Yakhini](https://zohary.cswp.cs.technion.ac.il/) and [Prof. Yacov Hel-Or](https://faculty.idc.ac.il/toky/) from the Interdisciplinary Center Herzliya. -->

## Abstract
 
Autoencoders represent an effective approach for computing the underlying factors characterizing datasets of different types. The latent representation of autoencoders have been studied in the context of enabling interpolation between data points by decoding convex combinations of latent vectors. This interpolation, however, often leads to artifacts or produces unrealistic results during reconstruction. 

We argue that these incongruities are due to the structure of the latent space and because such naively interpolated latent vectors deviate from the data manifold. 

In this work, we propose a regularization technique that shapes the latent representation to follow a manifold that is consistent with the training images and that drives the manifold to be smooth and locally convex. This regularization not only enables faithful interpolation between data points, as we show herein, but can also be used as a general regularization technique to avoid overfitting or to produce new samples for data augmentation.

## Motivation

1. **Autoencoder latent spaces are non-convex**: While they represent an effective approach for exposing latent factors, autoencoders demonstrate visible artifacts while interpolating a convex sum of latent vectors.
2. **GANs are not bidirectional**: To interpolate between two real data points, we must map the datapoints back into latent space where admissible interpolation can be performed. Such inverse mapping is not a part of the GAN framework. Additionally, the latent space of the GAN does not necessarily encode a smooth parameterization of the data. 

## Manifold Data Interpolation

Before presenting the proposed approach we would like to define what constitutes a proper interpolation between two data points. There are many possible paths between two points on the manifold. Even if we require the interpolations to be on a geodesic path, there might be infinitely many such paths between two points. Therefore, we relax the geodesic requirement and define less restrictive conditions.

Formally, assume we are given a dataset sampled from a target domain \\(\cal{X}\\). We are interested in interpolating between two data points \\( \boldsymbol x\_i \\) and \\( \boldsymbol x\_j \\) from \\(\cal{X}\\). Let the interpolated points be \\( \hat{\boldsymbol{x}}\_{i \rightarrow j}(\alpha) \\) for \\( \alpha \in [0,1] \\) and let \\(P(\boldsymbol x)\\) be the probability that a data point \\(\boldsymbol x\\) belongs to \\(\cal{X}\\).  

We define an interpolation to be an **admissible interpolation** if \\( \hat{\boldsymbol{x}}\_{i \rightarrow j}(\alpha) \\) satisfies the following conditions:

1. **Boundary conditions**: \\( \hat{\boldsymbol{x}}\_{i \rightarrow j}(0) = \boldsymbol x\_i \\) and \\( \hat{\boldsymbol{x}}\_{i \rightarrow j}(1) = \boldsymbol x\_j \\)

2. **Monotonicity**: We require that under some defined distance on the manifold \\( d(\boldsymbol x,\boldsymbol x') \\) the interpolated points will depart from \\( \boldsymbol x\_i \\) and approach \\( \boldsymbol x\_j \\), as the parameterization \\( \alpha \\) goes from \\(0\\) to \\(1\\). Namely, \\( \forall \alpha' \geq \alpha \\):

    \\[ d(\hat{\boldsymbol{x}}\_{i \rightarrow j}(\alpha), \boldsymbol x\_i ) \leq d(\hat{\boldsymbol{x}}\_{i \rightarrow j}(\alpha'),\boldsymbol x\_i) \\]

    and similarly:

    \\[ d( \hat{\boldsymbol{x}}\_{i \rightarrow j}(\alpha'), \boldsymbol x\_j ) \leq d(\hat{\boldsymbol{x}}\_{i \rightarrow j}(\alpha),\boldsymbol x\_j) \\]

3. **Smoothness**: The interpolation function is Lipschitz continuous with a constant \\(K\\): 

    \\[ \|\| \hat{\boldsymbol{x}}\_{i \rightarrow j}(\alpha), \hat{\boldsymbol{x}}\_{i \rightarrow j}(\alpha+t) \|\| \leq K \| t \| \\]

4. **Credability**: We require that \\( \forall \alpha \in [0,1] \\) it is highly probable that interpolated images, \\( \hat{\boldsymbol{x}}\_{i \rightarrow j}(\alpha) \\) belong to \\(\cal{X} \\). Namely, 

    \\[ P(\hat{\boldsymbol{x}}\_{i \rightarrow j}(\alpha)) \geq 1-\beta \mbox{ for some constant \\(\beta \geq 0\\)}  \\]

## Proposed Approach

Following the above definitions for an admissible interpolation, we propose a new approach, called **Autoencoder Adversarial Interpolation** (AEAI), which shapes the latent space according to the above requirements.

For pairs of input data points \\( \boldsymbol{x}\_i, \boldsymbol{x}\_j\\), we linearly interpolate between them in the latent space: \\( \boldsymbol{z}\_{i \rightarrow j}(\alpha) = (1-\alpha) \boldsymbol{z}\_i + \alpha \boldsymbol{z}\_j \\), where \\( \alpha \in [0,1] \\).

1. The first loss term \\( {\cal L}\_R \\) is a standard reconstruction loss and is calculated for the two endpoints \\( \boldsymbol{x}\_i \\) and \\( \boldsymbol{x}\_j \\): 

    \\[ \cal{L}\_{R}^{i \rightarrow j} = \cal{L}(\boldsymbol{x}\_i,\hat{\boldsymbol{x}}\_i) + \cal{L}(\boldsymbol{x}\_j,\hat{\boldsymbol{x}}\_j) \\]
    where \\({\cal L}(\cdot,\cdot)\\) is some loss function between the two images.

2. We use a discriminator \\( D(\boldsymbol{x})\\) to differentiate between real and interpolated data points to encourage the network to fool the discriminator so that interpolated images are indistinguishable from the data in the target domain \\(\cal{X}\\).

    \\[ \cal{L}\_A^{i \rightarrow j}= \sum\_{n=0}^{M} -\log D(\hat{\boldsymbol{x}}\_{i \rightarrow j}(n/M)) \\]

3. The cycle-consistency loss \\(\cal{L}\_C\\) encourages the encoder and the decoder to produce a bijective mapping:
    \\[ \cal{L}\_{C}^{i \rightarrow j}= \sum_{n=0}^{M} \| \boldsymbol{z}\_{i \rightarrow j}(n/M)- \hat{\boldsymbol{z}}\_{i \rightarrow j}(n/M)\|^2 \\]

    where \\(\hat{\boldsymbol{z}}\_{i \rightarrow j}(\alpha) =f(g(\boldsymbol{z}\_{i \rightarrow j}(\alpha))) \\). 

4. The last term \\(\cal{L}\_S\\) is the smoothness loss encouraging \\(\hat{\boldsymbol{x}}(\alpha)\\) to produce smoothly varying interpolated points between \\( \boldsymbol{x}_i \\) and \\( \boldsymbol{x}_j\\):
    \\[ \cal{L}\_{S}^{i \rightarrow j}= \sum\_{n=0}^M \left \| {\frac{\partial \hat{\boldsymbol{x}}\_{i \rightarrow j}(\alpha) }{\partial \alpha}}  \right \|^2\_{\alpha={n}/M} \\]

Putting everything together we define the loss \\(\cal{L}\_{i \rightarrow j}\\) between pairs \\( \boldsymbol{x}_i \\) and \\( \boldsymbol{x}_j\\) as follows: 
\\[ \cal{L}^{i \rightarrow j} = \cal{L}\_R^{i \rightarrow j} + \lambda\_A \cal{L}\_A^{i \rightarrow j} + \lambda\_C \cal{L}\_C^{i \rightarrow j} + \lambda\_S \cal{L}\_S^{i \rightarrow j} \\]
 where \\(\cal{L}\_R, \cal{L}\_A, \cal{L}\_C, \cal{L}\_S\\) are the reconstruction, adversarial, cycle, and smoothness losses, respectively.



<figure>
  <img width="1500" src="{{site.baseurl | prepend: site.url}}images/latent_intuition.png" alt="latent intuition"/>
</figure>
Data interpolation using AEAI. Two points \\(\boldsymbol{x}\_i, \boldsymbol{x}\_j\\) are located on the input data manifold (solid black line). The encoder \\( f(\x)\\) maps input points into the latent space \\(\boldsymbol{z}\_i\\), \\(\boldsymbol{z}\_j\\) (red arrows). Linear interpolation in the latent space is represented by the blue dashed line. The interpolated latent codes are mapped back into the input space by the decoder \\(g(\z)\\) (blue arrows). 

<img width="1500" alt="1" src="{{site.baseurl | prepend: site.url}}images/latent_intuition.png">
Data interpolation using AEAI. Two points \\(\boldsymbol{x}\_i, \boldsymbol{x}\_j\\) are located on the input data manifold (solid black line). The encoder \\( f(\x)\\) maps input points into the latent space \\(\boldsymbol{z}\_i\\), \\(\boldsymbol{z}\_j\\) (red arrows). Linear interpolation in the latent space is represented by the blue dashed line. The interpolated latent codes are mapped back into the input space by the decoder \\(g(\z)\\) (blue arrows). 


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