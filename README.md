# Deep Learning for Automatic Modulation Classification

We propose an efficient and lightweight convolutional neural network (CNN) for the task of automatic modulation classification (AMC). Before sending the received signal into our CNN model, we transform the signal to image domain with the proposed accumulated polar feature. It significantly improves the prediction accuracy and makes it more robust under fading channel. 
We provide the source code for the implementation of conventional approaches (Maximum Likelihood and Cumulant) and our deep learning based approaches. Hope this code is useful for peer researchers. If you use this code or parts of it in your research, please kindly cite our paper:

- **Related publication 1:** Chieh-Fang Teng, Ching-Chun Liao, Chun-Hsiang Chen, and An-Yeu (Andy) Wu, "[Polar Feature Based Deep Architectures for Automatic Modulation Classification Considering Channel Fading](https://ieeexplore.ieee.org/document/8646375)," *accepted by 2018 IEEE Global Conference on Signal and Information Processing (GlobalSIP).*

- **Related publication 2:** Chieh-Fang Teng, Ching-Yao Chou, Chun-Hsiang Chen, and An-Yeu (Andy) Wu, "[Accumulated Polar Feature-based Deep Learning for Efficient and Lightweight Automatic Modulation Classification with Channel Compensation Mechanism](https://arxiv.org/abs/2001.01395)," *arXiv:2001.01395, 2020.*
---

## Required Packages

- python 3.6.5
- numpy 1.16.4
- tensorflow 1.14.0
- keras 2.2.5
- Matlab R2017a

## Source Code
### Matlab
- Test_ML.m test the conventional likelohood-based approach of maximum likelihood (ML) and hybrid likelihood ratio test (HLRT)
  - Adjust the use of ML or HLRT
  - MaximunLikelihood.m
  - HybridLRT.m: set the preferred searching space for amplitude and phase

- Test_Cumu.m: test the conventional feature-based approach of cumulant
  - Cumulant.m: adjust the feature used for the classification

- AMC_raw.m: generate the training data for NN model and plot the figure in our paper
  - sig2pic.m: transform the received signal to image
  - sig2pic_accu.m: transform the received signal to image with historical information
  - sig2pic_gaussian.m: transform the received signal to image by Gaussian distribution

### Python
- Users need to customize the *config.py* and *Polar-NN-MULT.ipynb* as
  - `N` : Block length 
  - `K` : Information length
  - `ebn0` : Desired SNR range 
  - `numOfWord` : Desired batch size 
  - `bp_iter_num` : The number of iteration for BP
  - `RNN` : Whether using recurrent architecture (1 = yes)
  - `quantize_weight` : Different mechanism for weight quantization (0 for non-quantize, 1 for normal, 2 for binarized, 3 for bin, 4 for binarized bin)
  - `bin_bit` : The number of different value
  - `binary_prec` : The number of weight precision (binary_prec must >= bin_bit)

## Contact Information

   ```
Chieh-Fang Teng:
        + jeff@access.ee.ntu.edu.tw
        + d06943020@ntu.edu.tw
   ```
