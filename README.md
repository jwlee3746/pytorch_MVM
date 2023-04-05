# MVM+ : Generative Model via Adversarial Metric Learning
Follow-up study of "Manifold Matching via Deep Metric Learning for Generative Modeling" (ICCV 2021).
</p>
Origin: https://github.com/dzld00/pytorch-manifold-matching
</p>
Paper: https://arxiv.org/abs/2106.10777
</p>

# Objective functions
Objective for metric learning:
```
metric_loss = Cos_(ml_real_out,ml_real_out_shuffle,ml_fake_out_shuffle) 
```
Objective for manifold matching with learned metric:
```
g_loss = d_hausdorff_(ml_real_out, ml_fake_out) 
```

# Dataset
Download data to the data path. The sample code uses [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.

# Training
To train a model for unconditonal generation, run:

```
python train.py
python train_MNIST.py
python train_synthetic.py
```
&emsp;
&emsp;
&emsp;
&emsp;

<table>
  <tr>
      <td><img alt="GAN" src="/images/0209_0027_spiral_30000.gif">
      <img alt="MVM" src="/images/0209_0009_MVM_spiral_30000_.gif">
      <img alt="MVM+" src="/images/0208_1631_MVM++_spiral_30000_Sota재현.gif">
  <tr>
</table>

# Citation
```
@misc{daiandhang2021manifold,
      title={Manifold Matching via Deep Metric Learning for Generative Modeling}, 
      author={Mengyu Dai and Haibin Hang},
      year={2021},
      eprint={2106.10777},
      archivePrefix={arXiv}
}
```
