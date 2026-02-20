# bandit_approach_simulation
The aim of this project is to analyse and simulate the performance of the bandit approach algorithm to multiple testing cases, with simulated and real data.

## Overview

This repository implements adaptive bandit algorithms for sequential experimental design in megastudies, based on the NeurIPS 2018 paper *"A Bandit Approach to Sequential Experimental Design with False Discovery Control"* (Jamieson & Jain).

The goal is to **maximize the True Positive Rate (TPR)** while controlling the **False Discovery Rate (FDR)** at level δ, using as few samples as possible.
