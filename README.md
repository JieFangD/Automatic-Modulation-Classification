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
- Test_ML.m: test the conventional likelohood-based approach of maximum likelihood (ML) and hybrid likelihood ratio test (HLRT)
  - Adjust the use of ML or HLRT
  - MaximunLikelihood.m
  - HybridLRT.m: set the preferred searching space for amplitude and phase

- Test_Cumu.m: test the conventional feature-based approach of cumulant
  - Cumulant.m: adjust the feature used for the classification

- AMC_raw.m: generate the training data for NN model and visualize the figure used in our paper
  - sig2pic.m: transform the received signal to image
  - sig2pic_accu.m: transform the received signal to image with historical information
  - sig2pic_gaussian.m: transform the received signal to image by Gaussian distribution

### Python
- classify_cnn_pic.py: transform the received signal to image in I/Q domain and train CNN model
  - python3 classify_cnn_pic.py [training data] [testing data] [model size]
    - e.g.: python3 classify_cnn_pic.py ../Data/raw1000_1000_8.mat ../Data/raw500_1000_8.mat 2

- classify_cnnpolar_pic.py: transform the received signal to image in polar domain and train CNN model
  - Adjust whether using Gaussian distribution to generate image
  - python3 classify_cnnpolar_pic.py [training data] [testing data] [model size]
    - e.g.: python3 classify_cnnpolar_pic.py ../Data/raw1000_1000_8.mat ../Data/raw500_1000_8.mat 2

- util.py
  - Adjust the generated image with accumulated information or not

## Contact Information

   ```
Chieh-Fang Teng:
        + jeff@access.ee.ntu.edu.tw
        + d06943020@ntu.edu.tw
   ```
