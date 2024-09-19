# Generative Models Implementation

This repository contains implementations of various generative models, including CycleGAN, pix2pix, SNGAN, StyleGAN2, UNET, VAE, and WGAN. Each model is implemented in a Jupyter Notebook, organized into separate folders for easy navigation.

## Repository Structure

- **CycleGAN/**
  - Contains the Jupyter notebook implementing CycleGAN. CycleGAN is used for image-to-image translation without requiring paired examples.
  
- **pix2pix/**
  - Contains the Jupyter notebook implementing pix2pix. Pix2pix is another image-to-image translation model, but it requires paired examples during training.

- **SNGAN/**
  - Contains the Jupyter notebook implementing Spectral Normalization GAN (SNGAN). SNGAN is a variant of GANs that stabilizes training by applying spectral normalization to the generator and discriminator.

- **StyleGAN2/**
  - Contains the Jupyter notebook implementing StyleGAN2. StyleGAN2 is an advanced generative model known for high-quality and fine-grained control over image generation.

- **UNET/**
  - Contains the Jupyter notebook implementing U-Net. U-Net is primarily used for image segmentation tasks, but can also be used in some generative tasks such as conditional image generation.

- **VAE/**
  - Contains the Jupyter notebook implementing Variational Autoencoder (VAE). VAE is a generative model that encodes input data into a latent space and generates new data samples by sampling from this latent space.

- **WGAN/**
  - Contains the Jupyter notebook implementing Wasserstein GAN (WGAN). WGAN improves upon traditional GANs by using the Wasserstein distance to make the training process more stable.

## Requirements

To run the notebooks in this repository, you'll need to have the following installed:

- Python 3.x
- Jupyter Notebook
- TensorFlow or PyTorch (depending on the model)
- Numpy
- Matplotlib
- Scikit-learn
- Pillow (for image processing)


## References
Here are the primary papers and references for each of the models implemented in this repository:

- **CycleGAN:**
  - Zhu, Jun-Yan, et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.
  - https://arxiv.org/abs/1703.10593

- **pix2pix:**
  - Isola, Phillip, et al. "Image-to-Image Translation with Conditional Adversarial Networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
  - https://arxiv.org/abs/1611.07004

- **SNGAN:**
  - Miyato, Takeru, et al. "Spectral Normalization for Generative Adversarial Networks." International Conference on Learning Representations (ICLR), 2018.
  - https://arxiv.org/abs/1802.05957

- **StyleGAN2:**
  - Karras, Tero, et al. "Analyzing and Improving the Image Quality of StyleGAN." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
  - https://arxiv.org/abs/1912.04958

- **UNET:**
  - Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention       (MICCAI), 2015.
  - https://arxiv.org/abs/1505.04597

- **VAE:**
  - Kingma, Diederik P., and Max Welling. "Auto-Encoding Variational Bayes." International Conference on Learning Representations (ICLR), 2014.
  - https://arxiv.org/abs/1312.6114

- **WGAN:**
  - Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein GAN." Proceedings of the 34th International Conference on Machine Learning (ICML), 2017.
  - https://arxiv.org/abs/1701.07875

Feel free to explore each folder for detailed implementations of these models. Happy coding!
