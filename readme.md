# Promptable Counterfactual Diffusion Model for Brain Tumor Analysis with MRIs

This repository contains the implementation of the "Promptable Counterfactual Diffusion Model for Brain Tumor Analysis with MRIs."

## Key Features

- **Counterfactual Generation**: ­Developed a unified solution for brain tumor segmentation and generation using a novel diffusion model with mask-level prompting for guided manipulation of MRI images.
- **Transformer Integration**: ­Integrate Transformer-based denoising network to capture global context more effectively, enhancing the model's performance in handling complex tumor morphologies. 
- **Advanced Tumor Synthesis and Generation**: Implemented a two-step tumor generation process, allowing for realistic tumor structure synthesis and position transfer, enhancing data augmentation and clinical decision support.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/arcadelab/counterfactual_diffusion.git
cd counterfactual_diffusion
pip install -r requirements.txt
