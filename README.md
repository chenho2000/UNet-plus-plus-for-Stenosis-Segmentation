# UNet++ for Stenosis Segmentation

## Overview
This repository contains the implementation and experiments for the paper **"A Nested U-Net Architecture for Stenosis Segmentation"**. The project focuses on improving stenosis detection in coronary artery disease (CAD) using the UNet++ architecture, which enhances the original U-Net by introducing nested skip pathways and deep supervision. The work is part of the CS 679 Final Project at the University of Waterloo.

## Project Structure
- **Report**: Detailed documentation of the methodology, experiments, and results (`report.pdf`).
- **Source Code**: Implementation of UNet++, U-Net, and Wide U-Net models, along with additional experiments (e.g., ResNet-style blocks).
- **Dataset**: Information about the ARCADE dataset used for stenosis detection (X-ray coronary angiography images).

## Key Features
- **UNet++ Architecture**: Nested skip pathways bridge the semantic gap between encoder and decoder, improving segmentation accuracy.
- **Deep Supervision**: Combines Binary Cross-Entropy (BCE) and Dice loss for enhanced training and model pruning.
- **Comparative Analysis**: Evaluates UNet++ against U-Net and Wide U-Net, demonstrating superior performance in stenosis detection.
- **Model Pruning**: Explores trade-offs between inference speed and accuracy.

## Results
| Architecture    | Parameters | F1 Score | IoU    |
|-----------------|------------|----------|--------|
| U-Net          | 7.77M      | 49.36    | 36.76  |
| Wide U-Net     | 9.29M      | 53.52    | 39.37  |
| UNet++ (w/o DS)| 9.16M      | 57.47    | 41.30  |
| UNet++ (w/ DS) | 9.16M      | 57.76    | 41.68  |
