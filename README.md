# DA-SpMM LightGBM Module
This is the classifier for DA-SpMM based on LightGBM

By runing `main.py`, you will be able to train a LightGBM prediction model.

The input of the model is a vector with 4 entries, namely 'nnz', 'mat_size', 'std_row','N'.
If you activate the `hardware-aware` mode in `main.py`, the input of the model will include 3 hardware features.

The output of the model is a tag (from 0 to 7), corresponding to the 8 different algorithms mentioned in our paper.

In files end with .csv, we provide some sample data for the project, by runing main.py directly, you will get to know more about the project.

Please find our DAC'22 paper [here](https://arxiv.org/abs/2202.08556).
