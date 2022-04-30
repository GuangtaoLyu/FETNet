# FETNet
FETNet: Feature Erasing and Transferring Network for Scene Text Removal

![avatar](./images/FETNet.png)

### Dataset

This Flickr-ST dataset is a real world dataset\cite{Scene Text Eraser} including 3,004 images with 2,204 images for training and 800 images for testing. Scene text in Flickr-ST has arbitrary orientation and shape. It provides five types annotations, text removed images (b), pixel-level text masks (c), character instance segmentation labels, category labels and character-level bounding box labels(d). The word-level scene text regions can be calculated simply Implicitly from those labels. To the best of our knowledge, Flickr-ST is the only dataset with such comprehensive annotations for scene text related tasks.

![avatar](./images/annotation.png)

Yunpan : (link:  https://pan.baidu.com/s/1K3l_CQL3LjKxGvk4fayxkw  password: nofd) (Size = 701MB).

Google Driver: (link: https://drive.google.com/file/d/1gUWkteBwgFGvY7z2sqAB96Dgjy8KqtwD/view?usp=sharing) (Size = 701MB).



### Implementation details

We train FETNet on the training sets of SCUT-EnsText \cite{EnsNet}, SCUT-Syn and Flickr-ST, and evaluate them on their corresponding testing sets, respectively. The masks are generated through the subtraction from the input images and the corresponding labels. We follow  \cite{EraseNet} to apply data augmentation during training. The model is optimized using the Adam algorithm. The initial learning rate of generator is set to be 0.001, and the discriminator is 0.002. The learning rate decayed by 50\% every 8 epochs. Following the training procedures of GAN, we alternately train the generator and discriminator in a single NVIDIA GPU with batch size of 6 and input image size of 256x256.

### Requirements

- PyTorch == 1.7.0(1.x)
- CUDA 10.1 

### Train

```python
python run.py  --text_root ./dataset/train/img --mask_root ./dataset/train/mask --gt_root ./dataset/train/inpaint --result_save_path ./results/xxx  
```

### Test

```python
python run.py  --text_root ./dataset/test/img --mask_root ./dataset/test/mask --gt_root ./dataset/test/inpaint --model_path_g ./checkpoint/xxx.pth  --model_path_d ./checkpoint/xxx.pth --result_save_path ./results/xxx  --test
```

### Evaluation

```python
python evaluation.py  
```



### References

[1]T. Nakamura, A. Zhu, K. Yanai, S. Uchida, Scene text eraser, 2017.

[2]C. Liu, Y. Liu, L. Jin, S. Zhang, Y. Wang, Erasenet: End-to-end text removal in the wild, IEEE Transactions on Image Processing PP (99) (2020) 1–1.

[3]S. Zhang, Y. Liu, L. Jin, Y. Huang, S. Lai, Ensnet: Ensconce text in the wild, in: AAAI, 2019.

