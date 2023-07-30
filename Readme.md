# Finding Efficient Pruned Network via Refined Gradients for Pruned Weights (ACM MM 2023)

This repository is the official implementation of [Finding Efficient Pruned Network via Refined Gradients for Pruned Weights](https://arxiv.org/pdf/2109.04660.pdf). 


## Requirements

To install requirements using [environment.yml](environment.yml) refer to the [documentation.](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file


## Training

[run_dcil.py](run_dcil) is the code for training  **with DCIL**. 


> run the dcil_res32.sh for training resnet-32 with DCIL.

## Citation
Please refer to the following citation if this work is useful for your research.

### Bibtex:

```
@article{kim2021dynamic,
  title={Dynamic collective intelligence learning: Finding efficient sparse model via refined gradients for pruned weights},
  author={Kim, Jangho and Yoo, Jayeon and Song, Yeji and Yoo, KiYoon and Kwak, Nojun},
  journal={arXiv preprint arXiv:2109.04660},
  year={2021}
}
```