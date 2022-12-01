# Identification of Rare Cortical Foling Patterns using Unsupervised Deep Learning

Official Pytorch implementation for https://arxiv.org/abs/2211.16213.
The project aims to identify rare cortical folding patterns in the central sulcus region thanks to unsupervised deep learning.

![image](/images/graphical_abstract.pdf)

## Data 
Data are available in the [data](./data) directory.



## Configuration
First, you need to update `config.py` with:
- your directories
- your input data dimensions

```
self.data_dir = "/path/to/data/directory"
self.subject_dir = "/path/to/list_of_subjects"
self.save_dir = "/path/to/saving/directory"

self.in_shape = (c, h, w, d)
```