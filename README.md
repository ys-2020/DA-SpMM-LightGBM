# DA-SpMM LightGBM Module
This is the classifier for DA-SpMM based on LightGBM

By runing `main.py`, you will be able to train a LightGBM prediction model.

The input of the model is a vector with 4 entries, namely `nnz`, `mat_size`, `std_row`, `N`.
If you activate the `hardware-aware` mode in `main.py`, the input of the model will include 3 extra hardware features.

The output of the model is a tag (from 0 to 7), corresponding to the 8 different algorithms mentioned in our paper.

The corresponding relationships are as follows (Please refer to our [paper](https://arxiv.org/abs/2202.08556) for the exact definition of the algorithms) :

| Tag                 | Algorithm                   |
| ------------------- | --------------------------- |
| 0                   | RB_SR_RM                    |
| 1                   | RB_PR_RM                    |
| 2                   | EB_SR_RM                    |
| 3                   | EB_PR_RM                    |
| 4                   | RB_SR_CM                    |
| 5                   | RB_PR_CM                    |
| 6                   | EB_SR_CM                    |
| 7                   | EB_PR_CM                    |

In the folder `data`, we provide some sample data for the project, by runing main.py directly, you will get to know more about the project.

Please find our DAC'22 paper [here](https://arxiv.org/abs/2202.08556).

Please install [scikit-learn](https://scikit-learn.org/stable/) and [LightGBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) before running our repo. 
