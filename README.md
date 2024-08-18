# MFPD
[RESIDE数据集](https://pan.baidu.com/share/init?surl=NqAaec3MFwFU9ZM2lfR_4w) 提取码：v8ku 

[SOTS数据集](https://drive.google.com/open?id=1qZlnJN4ybjunc2BGh6kjOUfFdVxuNS-P)

我训练的[模型](https://pan.baidu.com/s/1xlfk_FNEDJimDQlk8FLn1Q) 提取码：1234

## 摘要
图像去雾是计算机视觉中的关键问题，基于深度学习的端到端去雾方法已成为该领域的主流方法，并且取得了不
错的效果。然而，这类方法对于高频信息的捕获上仍存在不足，导致图像清晰度和真实感降低。针对这一问题，本文提出了
一种基于多模态特征感知的去冗余去雾网络，通过设计多模态特征感知模型和多模态去冗余模型，增强了网络对高频信息的
捕获和利用能力。具体的，多模态特征感知模型包括高频信息感知模块和高频特征编码器。高频信息感知模块首先利用快速
傅里叶变换（FFT）将图像从空间域转换到频域，加强对图片的边缘、纹理和细节信息的捕获。然后，高频特征编码器利用
多尺度处理深入分析这些高频分量，生成丰富且表达力强的特征，使网络能够同时关注大尺度结构和小尺度细节信息。最后，
多模态去冗余模型引入通道注意力机制，自适应地调整特征权重，突出关键高频信息，减少冗余与噪声，从而在融合过程中
最大限度地保留图像细节，提升去雾效果的质量。为了验证其有效性，与 12 种经典的图像去雾算法进行了对比实验。在 S
OTS 数据集上，我们的平均峰值信噪比达到了 34.77dB，相比于性能第二的模型提升了 2.9%。


@conference{MSBDN-DFF,
		author = {Hang, Dong and Jinshan, Pan and Zhe, Hu and Xiang, Lei and Xinyi, Zhang and Fei, Wang and Ming-Hsuan, Yang},
		title = {Multi-Scale Boosted Dehazing Network with Dense Feature Fusion},
		booktitle = {CVPR},
		year = {2020}
	}
