# Semi-Online Knowledge Distillation

Implementations of SOKD.



# Requirements

This repo was tested with Python 3.8, PyTorch 1.5.1, torchvision 0.6.1, CUDA 10.1.

# Training

1. Train vanilla model by:

   ```python
   python main.py -c ./configs/vanilla.yaml --gpu 0 --name experimental_name
   ```

2. Train SOKD by:

   ```python
   python main.py -c ./configs/sokd.yaml --gpu 0 --name experimental_name
   ```




Compared methods can be found at the following repos:

[Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo)

[RepDistiller](https://github.com/HobbitLong/RepDistiller)