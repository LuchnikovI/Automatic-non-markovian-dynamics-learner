# Probing non-Markovian quantum dynamics with data-driven analysis:Beyond “black-box” machine-learning models
This is a repo accompanying the following paper [https://arxiv.org/abs/2103.14490](https://arxiv.org/abs/2103.14490)
(this arxiv preprint is not up to date, will be updated soon)

## Introduction

In this repo, we present code for a novel method of data-driven non-Markovian dynamics identification.
The new method allows one to build a model of a quantum system and its quantum environment based only on observed
non-markovian dynamics of the system. The main distinguishing features of the method include
- The ability of the method to identifie the effective dimension of the environment automatically from data
- Data efficiency and robustness to noise caused by the automatic model selection
- The ability of the method to extract spectral characteristics not only of the system but also of the environment
- The ability of the method to denoise tomographic data

## Content
- [simple_example.py](/simple_example.py): This code demonstrates how the framework lying under the hood of the numerical experiments works (one can think of it as quick_start file).
- [benchmarking.py](/benchmarking.py): This code runs benchmarking experiments comparing the proposed mthod with Transfer-Tensor method in terms of prediction accuracy.
The results of experiments are saved in benchmarking_results.pickle, the parameters of experiments are saved in benchmarking_parameters.pickle.
- [benchmarking_plotting.py](/benchmarking_plotting.py): This code plots the results of benchmarking experiments and save the corresponding pdf file in the root.
- [finite_de.py](/finite_de.py): This code runs experiments necessary for presentation of main features of the proposed method. It saves results in finite_de_results.pickle,
and experiments parameters in finite_de_parameters.pickle. The results of this file execution are used accros multiple plotters with names ``..._plotting.py''.
- [plotting_all.py](/plotting_all.py): This code plots all the figures at once and save corresponding pdf files in the root.

## Dependencies

- [JAX](https://github.com/google/jax), [Chex](https://github.com/deepmind/chex).

## Citation

We kindly ask you to cite our paper if you use our code/method:

@article{luchnikov2021probing,
  title={Probing non-Markovian quantum dynamics with data-driven analysis: Beyond" black-box" machine learning models},
  author={Luchnikov, IA and Kiktenko, EO and Gavreev, MA and Ouerdane, H and Filippov, SN and Fedorov, AK},
  journal={arXiv preprint arXiv:2103.14490},
  year={2021}
}
