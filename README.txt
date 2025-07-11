# GNN-Based Antibody-Antigen Binding Affinity Prediction

This repository contains code for preprocessing structural antibody-antigen complex data into graph format suitable for Graph Neural Network (GNN) modeling. The goal of this project is to develop a predictive model for estimating binding affinity based on 3D structural information.

## Overview

The core script (`gnn.py`) parses Protein Data Bank (PDB) files to extract chain metadata, residue-level features, and spatial relationships between amino acids. These are converted into graph structures using PyTorch Geometric for downstream GNN training.

This project is in progress and forms part of a larger effort to apply machine learning to antibody-antigen interaction modeling.

## Features

- Parses REMARK 5 metadata to identify antibody heavy/light and antigen chains  
- Extracts alpha-carbon coordinates and residue features (e.g., hydrophobicity)  
- Computes pairwise spatial proximity to form edge lists  
- Generates PyTorch Geometric `Data` objects for each structure  

## Requirements

- Python 3.8+  
- Biopython  
- NumPy  
- Pandas  
- PyTorch  
- PyTorch Geometric  

All dependencies are listed in `requirements.txt`.

## Getting Started

Clone the repo and install dependencies:

```bash
git clone https://github.com/edwinovna/gnnproject.git
cd gnn_project
pip install -r requirements.txt


Note: 'gnn.py' is still under development. Future updates will include model training, evaulation and prediction.

## Author

**Katerina Johnson**  
M.S. Biomedical Engineering (Bioinformatics concentration)  
University of New Mexico  
Expected Graduation: July 2025
