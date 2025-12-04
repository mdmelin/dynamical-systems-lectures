# dynamical-systems-lectures
### Run in colab
Todo
### Installation (for running locally)
Create a conda environment and activate it
```
conda create -n ssm-demo python=3.10
conda activate ssm-demo
pip install jupyter ipykernel numpy cython seaborn matplotlib
USE_OPENMP=True pip install git+https://github.com/lindermanlab/ssm.git --no-build-isolation
```
Then launch the jupyter notebook
```
jupyter notebook 
```
