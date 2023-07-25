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

4. To remove the example patient in `data/patient` and create the model directories, you can use the script `initialize.sh`. Simply run the following command:

```bash
bash initialize.sh
```

This will remove the example patient data and create the necessary model directories, such as `FastRCNN` and `ResNet`. 

Alternatively, if you prefer to do it manually, follow these steps:

1. Delete the example patient data located in the `data/patient` directory.

2. Create the model directories `FastRCNN` and `ResNet` in the `models` directory.

Now you can proceed with your project using the modified data and model directories.

## Usage

To use Malaria-NET for cell detection and classification, follow these steps:

1. Place patient folders with their respective images inside the 'data' folder. The folder structure should be as follows:

```
|-- data/
    |-- P001/
        |-- img1.png
        |-- img2.png
            .
            .
            .
        |-- imgk.png
    |-- P002/
    |-- P003/
        .
        .
        .
    |-- P00N/
```

2. Store Faster R-CNN and ResNet models inside the 'models' folder.

3. Run the Python script as follows: ```python malarianet.py -nr [name of the results]```

4. The results will be saved in a .csv file inside the 'results' folder.

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
    |-- __init__.py        # Use src as a module
    |-- functions.py       # Utility functions for data processing and visualization
|   |-- resnet.py          # ResNet-50 model implementation
|   |-- fastrcnn.py        # Faster R-CNN model implementation
|   |-- process_image.py   # Script for image processing in Malaria-NET
|-- README.md              # Project documentation and instructions
|-- LICENSE                # License information
```

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please create a pull request or open an issue on the GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.