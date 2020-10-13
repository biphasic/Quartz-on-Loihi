#!/bin/zsh
for run in {1..50}; do SLURM=1 ipython 13_CNN.ipynb; done
