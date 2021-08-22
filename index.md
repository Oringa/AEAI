# AEAI

The **Autoencoder Adversarial Interpolation** (AEAI) is a novel approach for the generation of realistic interpolation between manifold data. 

This paper was written under the supervision of [Prof. Zohar Yakhini](https://zohary.cswp.cs.technion.ac.il/) and [Prof. Yacov Hel-Or](https://faculty.idc.ac.il/toky/) from the Interdisciplinary Center Herzliya.

## Abstract
 
Autoencoders represent an effective approach for computing the underlying factors characterizing datasets of different types. The latent representation of autoencoders have been studied in the context of enabling interpolation between data points by decoding convex combinations of latent vectors. This interpolation, however, often leads to artifacts or produces unrealistic results during reconstruction. We argue that these incongruities are due to the structure of the latent space and because such naively interpolated latent vectors deviate from the data manifold. In this paper, we propose a regularization technique that shapes the latent representation to follow a manifold that is consistent with the training images and that drives the manifold to be smooth and locally convex. This regularization not only enables faithful interpolation between data points, as we show herein, but can also be used as a general regularization technique to avoid overfitting or to produce new samples for data augmentation.

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




<!-- |<img width="1604" alt="1" src="{{site.baseurl | prepend: site.url}}animations/1.gif"> | <img width="1604" alt="1" src="{{site.baseurl | prepend: site.url}}animations/1.gif"> | <img width="1604" alt="1" src="{{site.baseurl | prepend: site.url}}animations/1.gif"> | <img width="1604" alt="1" src="{{site.baseurl | prepend: site.url}}animations/1.gif"> | <img width="1604" alt="1" src="{{site.baseurl | prepend: site.url}}animations/1.gif"> | <img width="1604" alt="1" src="{{site.baseurl | prepend: site.url}}animations/1.gif"> |

|<img width="1604" alt="1" src="{{site.baseurl | prepend: site.url}}animations/1.gif"> | <img width="1604" alt="1" src="{{site.baseurl | prepend: site.url}}animations/1.gif"> | <img width="1604" alt="1" src="{{site.baseurl | prepend: site.url}}animations/1.gif"> |<img width="1604" alt="1" src="{{site.baseurl | prepend: site.url}}animations/1.gif"> | <img width="1604" alt="1" src="{{site.baseurl | prepend: site.url}}animations/1.gif"> | <img width="1604" alt="1" src="{{site.baseurl | prepend: site.url}}animations/1.gif"> |
 -->
<!-- <img width="1000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/aeai_chess.gif"> -->
<!-- <img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/aeai_chess.gif">
<img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/aeai_chess.gif">
<img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/aeai_chess.gif">
<img width="2000" alt="1" src="{{site.baseurl | prepend: site.url}}animations/aeai_chess.gif"> -->