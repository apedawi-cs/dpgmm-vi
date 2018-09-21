# Dirichlet process Gaussian mixture model (DPGMM)

In this repository, you will find a self-contained example of a Dirichlet process Gaussian mixture model (DPGMM) trained with variational inference in TensorFlow. For a description of the model and the approach for carrying out inference, please refer to [description.ipynb](https://github.com/apedawi-cs/dpgmm-vi/blob/master/description.ipynb). To run the script, make sure you have the requisite dependencies below, download [dpgmm_vi.py](https://github.com/apedawi-cs/dpgmm-vi/blob/master/dpgmm_vi.py) to the desired directly, and call `python dpgmm_vi.py` from it.

Below are some results:

![ELBO curve by iteration](https://github.com/apedawi-cs/dpgmm-vi/blob/master/elbo_curve.png)


### Python 2.7.10 dependencies
```
matplotlib==1.3.1
numpy==1.14.5
tensorflow==1.10.1
tensorflow_probability
```
