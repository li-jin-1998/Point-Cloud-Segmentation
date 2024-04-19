# [Point-Cloud-Segmentation](https://github.com/li-jin-1998/Point-Cloud-Segmentation)

Implementations of some point cloud segmentation methods.

## Project List

Below are implementations of several popular point cloud segmentation methods.

- [PointNet](https://github.com/charlesq34/pointnet): PointNet, developed by Charles Q. Allen, is a seminal work for
  processing point cloud data.
- [PointNet++](https://github.com/charlesq34/pointnet2): An upgraded version of PointNet, introducing more sophisticated
  local feature learning modules.
- [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch): KPConv, developed by Hugues Thomas and based on
  PyTorch, is an efficient point cloud convolution method.
- [NanoFLANN](https://github.com/jlblancoc/nanoflann): NanoFLANN, developed by Joan Blanco, is a fast K-nearest neighbor
  search library commonly used in point cloud processing.
- [ConvPoint](https://github.com/aboulch/ConvPoint): ConvPoint, developed by Alexandre Boulch, proposes convolution
  methods based on spherical harmonics.
- [PCNN-Tensorflow](https://github.com/matanatz/pcnn): A TensorFlow implementation of PCNN by Matanatz, which is a
  graph-based point cloud classification method.
- [Open-points](https://github.com/guochengqian/openpoints): OpenPoints, developed by Guocheng Qian, aims to perform
  point cloud segmentation through multi-scale feature learning.
- [PointNeXt](https://github.com/guochengqian/PointNeXt): PointNeXt, also developed by Guocheng Qian, further advances
  the PointNet series by introducing new modules to enhance performance.
- [Torch-Points3D](https://github.com/nicolas-chaulet/torch-points3d): Torch-Points3D, developed by Nicolas Chaulet, is
  a comprehensive framework for point cloud processing and segmentation.
- [PointCNN](https://github.com/yangyanli/PointCNN): PointCNN, developed by Yangyan Li, is a point cloud segmentation
  method based on convolutional neural networks.
- [Spotr](https://github.com/mlvlab/spotr): Spotr, developed by MLV Lab, is a point cloud segmentation method based on
  attention mechanisms.

## References

- Nguyen, A.; Le, B. 3D Point Cloud Segmentation: A Survey. In 2013 6th IEEE Conference on Robotics, Automation and
  Mechatronics (RAM); IEEE: Manila, Philippines, 2013; pp 225–230. https://doi.org/10.1109/RAM.2013.6758588.

- Wu, Z.; Song, S.; Khosla, A.; Yu, F.; Zhang, L.; Tang, X.; Xiao, J. 3D ShapeNets: A Deep Representation for Volumetric
  Shapes; 2015; pp 1912–1920.

- Chengzhi Wu; Junwei Zheng; Julius Pfrommer; Jürgen Beyerer. Attention-Based Point Cloud Edge Sampling. arXiv March 26,
  2023. http://arxiv.org/abs/2302.14673 (accessed 2024-04-10).

- Boulch, A. ConvPoint: Continuous Convolutions for Point Cloud Processing. arXiv February 19,
  2020. http://arxiv.org/abs/1904.02375 (accessed 2024-03-28).

- Guo, Y.; Wang, H.; Hu, Q.; Liu, H.; Liu, L.; Bennamoun, M. Deep Learning for 3D Point Clouds: A Survey. arXiv June 23,
  2020. http://arxiv.org/abs/1912.12033 (accessed 2024-03-20).

- Thomas, H.; Qi, C. R.; Deschaud, J.-E.; Marcotegui, B.; Goulette, F.; Guibas, L. J. KPConv: Flexible and Deformable
  Convolution for Point Clouds; 2019; pp 6411–6420.

- Ye, D.; Chen, W.; Zhou, Z.; Xie, Y.; Wang, Y.; Wang, P.; Foroosh, H. LidarMultiNet: Unifying LiDAR Semantic
  Segmentation, 3D Object Detection, and Panoptic Segmentation in a Single Multi-Task Network. arXiv June 23,
  2022. http://arxiv.org/abs/2206.11428 (accessed 2024-03-26).

- Hermosilla, P.; Ritschel, T.; Vázquez, P.-P.; Vinacua, À.; Ropinski, T. Monte Carlo Convolution for Learning on
  Non-Uniformly Sampled Point Clouds. ACM Trans. Graph. 2018, 37 (6), 235:1-235:
  12. https://doi.org/10.1145/3272127.3275110.

- Atzmon, M.; Maron, H.; Lipman, Y. Point Convolutional Neural Networks by Extension Operators. arXiv March 27,
  2018. https://doi.org/10.48550/arXiv.1803.10091.

- Zhao, H.; Jiang, L.; Jia, J.; Torr, P. H. S.; Koltun, V. Point Transformer; 2021; pp 16259–16268.

- Wu, W.; Qi, Z.; Fuxin, L. PointConv: Deep Convolutional Networks on 3D Point Clouds; 2019; pp 9621–9630.

- Qi, C. R.; Su, H.; Mo, K.; Guibas, L. J. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation;
  2017; pp 652–660.

- Qian, G.; Li, Y.; Peng, H.; Mai, J.; Hammoud, H. A. A. K.; Elhoseiny, M.; Ghanem, B. PointNeXt: Revisiting PointNet++
  with Improved Training and Scaling Strategies. arXiv October 12, 2022. http://arxiv.org/abs/2206.04670 (accessed
  2024-03-20).

- Vora, S.; Lang, A. H.; Helou, B.; Beijbom, O. PointPainting: Sequential Fusion for 3D Object Detection; 2020; pp
  4604–4612.

- Shi, S.; Wang, X.; Li, H. PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud; 2019; pp 770–779.

- Milioto, A.; Vizzo, I.; Behley, J.; Stachniss, C. RangeNet ++: Fast and Accurate LiDAR Semantic Segmentation. In 2019
  IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS); 2019; pp
  4213–4220. https://doi.org/10.1109/IROS40897.2019.8967762.

- Xu, J.; Zhang, R.; Dou, J.; Zhu, Y.; Sun, J.; Pu, S. RPVNet: A Deep and Efficient Range-Point-Voxel Fusion Network for
  LiDAR Point Cloud Segmentation; 2021; pp 16024–16033.

- Park, J.; Lee, S.; Kim, S.; Xiong, Y.; Kim, H. J. Self-Positioning Point-Based Transformer for Point Cloud
  Understanding. arXiv March 29, 2023. http://arxiv.org/abs/2303.16450 (accessed 2024-03-20).

- Maiti, A.; Elberink, S. O.; Vosselman, G. TransFusion: Multi-Modal Fusion Network for Semantic Segmentation. In
  Proceedings - 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, CVPRW 2023; IEEE, 2023;
  pp 6537–6547. https://doi.org/10.1109/CVPRW59228.2023.00695.

## Notes

- Make sure to check the latest versions and compatibility of each project, as libraries and frameworks may update over
  time.
- When using these resources, please adhere to the licensing agreements and copyright regulations of each project.

## Contribution Guidelines

Suggestions, bug reports, or additions of new resources are welcome. Please contribute by submitting pull requests or
creating new issues in the project issue tracker.

## Copyright

© 2023, Authors of Point-Cloud-Segmentation. This project is licensed under the MIT License.