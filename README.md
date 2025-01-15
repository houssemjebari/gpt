# GPT Training Pipeline



## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Welcome to the **GPT Training Pipeline**! This project provides a robust and scalable framework for training Generative Pre-trained Transformer (GPT) models. Leveraging advanced techniques such as Distributed Data Parallel (DDP) and a training pipeline that maintains clean and maintainable code architecture, while ensuring performance at the same time!

Whether you're a researcher aiming to push the boundaries of natural language processing or a developer integrating GPT models into applications, this pipeline offers the flexibility and performance you need.

## Project Structure

A well-organized project structure is crucial for scalability and maintainability. Here's an overview of the project's architecture:

    gpt/ 

        ├── main.py 
        ├── README.md 
        ├── requirements.txt 
        ├── setup.py 
        ├── configs/  
        |    └── train_config.py 
        |    └── gpt_config.py 
        |    └── lr_config.py 
        ├── training/ 
        │   ├── init.py 
        │   ├── trainer.py 
        │   ├── logger.py 
        │   └── evaluator.py 
        │   └── subject.py 
        │   └── observer.py 
        ├── utils/ 
        │   ├── init.py 
        │   ├── distributed.py 
        │   ├── helper.py 
        │   └── custom_scheduler.py 
        ├── models/ 
        │   ├── init.py 
        │   └── gpt.py │ 
        ├── data/ 
        │   ├── init.py 
        │   └── data_loader.py 
### Description of Key Directories and Files

- **`main.py`**: Entry point of the training pipeline. Initializes components and starts the training process.
- **`configs/`**: Contains configuration parameters for training, such as batch size, learning rates, and scheduler settings.
- **`training/`**: Houses the core training logic, including the Trainer class and observer implementations like Logger and Evaluator.
- **`utils/`**: Utility modules for distributed training, learning rate scheduling, and other helper functions.
- **`models/`**: Defines the  model architectures.
- **`data/`**: Handles data loading and preprocessing.

## Installation

Follow these steps to set up the GPT Training Pipeline on your machine:

### 1. Install Anaconda

Anaconda simplifies package management and deployment. If you haven't installed it yet, download and install [Anaconda](https://www.anaconda.com/products/distribution) for your operating system.

### 2. Create a Virtual Environment with Python 3.9.2

Open your terminal or command prompt and execute the following commands:

```bash
# Create a new conda environment named 'gpt_env' with Python 3.9.2
conda create -n gpt_env python=3.9.2

# Activate the environment
conda activate gpt_env
```
### 3. Install the project
```bash
pip install -e .
```

### 4. Ensure CUDA Compatibility for PyTorch
To leverage GPU acceleration, ensure that your system has CUDA installed and that it is compatible with the PyTorch version you intend to use.

Check CUDA Version to ensure that the CUDA version matches one supported by your desired PyTorch build.


```bash
nvcc --version
```
Visit the PyTorch Get Started page to find the appropriate installation command based on your CUDA version. For example, for CUDA 11.3:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch
```
### 5. Run the training pipeline
to run a specific pipeline modify the config file to get the desired model and the desired training pipeline configurations (e.g the number of steps, batch size, etc.)

Modify `configs/train_config.py` to set parameters like batch_size, learning_rate, warmup_steps, etc., according to your training requirements.


Modify `configs/gpt_config.py` to set parameters like number of layers, embedding size, etc., according to your model requirements.

Modify `configs/lr_config.py` to set parameters like number of warmup steps, number of max steps, etc., according to your training requirements.

Finally run this command.


```bash
python gpt/main.py
```