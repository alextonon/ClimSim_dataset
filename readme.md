# ClimSim Data Pipeline & Deep Learning Emulation
**ISAE-SUPAERO | SDD Deep Learning Class**

## 📌 Project Overview
This project focuses on the implementation of a deep learning pipeline designed to emulate atmospheric physics using the **ClimSim dataset** (https://arxiv.org/pdf/2306.08754). The goal of this paper is to bridge the gap between high-fidelity climate simulations and hybrid ML-physics models.

My work mostly revolves arround data processing, model inference, and evaluation of results of a personaly trained Multi-Layer Perceptron (MLP) model. This pipeline has been rewritten but is inspired by the original **ClimSim-Keras** codebase.

 I also tried **ERA5 reanalysis data**, interpolating it to the **E3SM model grid** (60 vertical levels), and performing inference using the pre trained MLP.


## 🚀 Getting Started

### Prerequisites
Warning : the number of libraries required is quite horrific, because of the torch and keras framework. If you don't want to run everything I can show you a proof that everything is working...
* Python 3.12+
* `xarray`, `numpy`, `pandas`, `matplotlib`, `torch`, `tensorflow`, `sklearn`, `h5netcdf`, `netcdf4`, `zarr`

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/alextonon/dataset.git](https://github.com/alextonon/dataset.git)
