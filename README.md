# NIC
End-to-End Learnt Image Compression via Non-Local Attention Optimization and Improved Context Modeling

[Vision Lab](vision.nju.edu.cn), Nanjing University

Tong Chen, Haojie Liu, Zhan Ma, Qiu Shen, Xun Cao and Yao Wang

## Abstract
This   paper   proposes   an   end-to-end   learnt   lossyimage  compression  approach  which  is  built  on  top  of  the  deepnerual   network   (DNN)-based   variational   auto-encoder   (VAE)structure  with  Non-Local  Attention  optimization  and  ImprovedContext  modeling  (NLAIC).  Our  NLAIC  1)  embeds  non-localnetwork   operations   as   non-linear   transforms   in   both   mainand  hyper  coders  for  deriving  respective  latent  features  andhyperpriors  by  exploiting  both  local  and  global  correlations,  2)applies attention mechanism to generate implicit masks that areused  to  weigh  the  features  for  adaptive  bit  allocation,  and  3)implements the improved conditional entropy modeling of latentfeatures  using  joint  3D  convolutional  neural  network  (CNN)-based autoregressive contexts and hyperpriors. Towards the prac-tical application, additional enhancements are also introduced tospeed  up  the  computational  processing  (e.g.,  parallel  3D  CNN-based  context  prediction),  decrease  the  memory  consumption(e.g., sparse non-local processing) and reduce the implementationcomplexity  (e.g.,  a  unified  model  for  variable  rates  without  re-training).  The  proposed  model  outperforms  existing  learnt  andconventional  (e.g.,  BPG,  JPEG2000,  JPEG)  image  compressionmethods,  on  both  Kodak  and  Tecnick  datasets  with  the  state-of-the-art  compression  efficiency,  for  both  PSNR  and  MS-SSIMdistortion  measurements.


## Materials
[Paper](https://arxiv.org/abs/1910.06244)

[Code & Pretrained Models](http://yun.nju.edu.cn/f/16ce608723/)
