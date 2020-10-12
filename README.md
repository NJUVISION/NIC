# NIC
End-to-End Learnt Image Compression via Non-Local Attention Optimization and Improved Context Modeling

[Vision Lab](vision.nju.edu.cn), Nanjing University

[Tong Chen](tongxyh.github.io), Haojie Liu, Zhan Ma, Qiu Shen, Xun Cao and Yao Wang

## Abstract
This paper proposes an end-to-end learnt lossy image compression approach which is built on top of the deep nerual network (DNN)-based variational auto-encoder (VAE)structure with  Non-Local Attention optimization and Improved Context modeling (NLAIC). Our NLAIC 1) embeds non-local network operations as non-linear transforms in both main and hyper coders for deriving respective latent features and hyperpriors by exploiting both local and global correlations, 2) applies attention mechanism to generate implicit masks that areused to weigh the features for adaptive bit allocation, and 3) implements the improved conditional entropy modeling of latent features using joint 3D convolutional neural network (CNN) based autoregressive contexts and hyperpriors. Towards the practical application, additional enhancements are also introduced to speed up the computational processing (e.g., parallel 3D CNN-based context prediction), decrease the memory consumption (e.g., sparse non-local processing) and reduce the implementation complexity (e.g., a unified model for variable rates without re-training). The proposed model outperforms existing learnt and conventional (e.g., BPG, JPEG2000, JPEG) image compression methods, on both Kodak and Tecnick datasets with the state-of-the-art compression efficiency, for both PSNR and MS-SSIM distortion measurements.


## Materials
[Paper](https://arxiv.org/abs/1910.06244)

[Code & Pretrained Models](http://yun.nju.edu.cn/f/16ce608723/)
