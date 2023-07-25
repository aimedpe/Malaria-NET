# Malaria-NET

Malaria-NET is a project aimed at detecting and classifying malaria-infected cells from microscopic images using deep learning techniques. The project uses a Faster R-CNN model with a ResNet-50 backbone for object detection and classification tasks.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Malaria is a life-threatening disease caused by Plasmodium parasites transmitted to humans through the bite of infected female Anopheles mosquitoes. Microscopic examination of blood smears is one of the standard methods used for malaria diagnosis. This project aims to automate this process by utilizing deep learning models for detecting and classifying malaria-infected cells.

## Dependencies

To run this project, you need the following dependencies:

- Python 3.x
- PyTorch
- torchvision
- numpy
- Pillow (PIL)
- OpenCV
- scikit-image

You can install the required packages by running:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/Malaria-NET.git
```

2. Change into the project directory:

```bash
cd Malaria-NET
```

3. Install the dependencies as mentioned in the [Dependencies](#dependencies) section.

## Usage

To use the Malaria-NET for cell detection and classification, follow these steps:

1. Prepare your dataset with annotated bounding boxes for the malaria-infected cells.

2. Train the Faster R-CNN model using the provided training script or load a pre-trained model.

3. Use the trained model for inference on new images to detect and classify malaria-infected cells.

For detailed usage instructions, refer to the documentation and example notebooks provided in the project.

## Project Structure

The project structure is organized as follows:

```
Malaria-NET/
|-- data/                  # Data directory (place your dataset here)
|-- models/                # Model directory (place your models here)
    |-- FastRCNN/          # Instances of FasterRCNN models for P.falciparum and P.vivax
    |-- ResNet/            # Instanes of Resnet50 models for P.falciparum, P.vivax and P.falciparum against P.vivax
|-- results/               # Results directory (results saved here)
|-- src/                   # Source code directory
|   |-- functions.py       # Utility functions for data processing and visualization
|   |-- resnet.py          # ResNet-50 model implementation
|   |-- fastrcnn.py        # Faster R-CNN model implementation
|   |-- train.py           # Script for training the model
|   |-- predict.py         # Script for making predictions using the trained model
|-- notebooks/             # Jupyter notebooks for demonstration and usage examples
|-- README.md              # Project documentation and instructions
|-- LICENSE                # License information
```

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please create a pull request or open an issue on the GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.