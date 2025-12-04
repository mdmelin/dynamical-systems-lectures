# dynamical-systems-lectures
### Run in colab
<a target="_blank"
   href="https://colab.research.google.com/github/mdmelin/dynamical-systems-lectures/">
  <img src="https://colab.research.google.com/assets/colab-badge.svg"
       alt="Open In Colab"/>
</a>
### Installation (for running locally)
Create a conda environment and activate it
```
conda create -n ssm-demo python=3.10
conda activate ssm-demo
git clone https://github.com/mdmelin/dynamical-systems-lectures.git
cd dynamical-systems-lectures
pip install -e .
USE_OPENMP=True pip install git+https://github.com/lindermanlab/ssm.git --no-build-isolation
```
Then launch the jupyter notebook
```
jupyter notebook 
```
