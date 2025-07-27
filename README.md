# ECT-BoFM

**📄 Paper:** [Edge- and Color–Texture-Aware Bag-of-Local-Features Model for Accurate and Interpretable Skin Lesion Diagnosis](https://www.mdpi.com/2075-4418/15/15/1883)

Code release for `Edge- and Color-texture-aware Bag-of-Local-Features Model for Accurate and Interpretable Skin Lesion Diagnosis`.

![enter image description here](https://github.com/Dichao-Liu/ECT-BoFM/blob/main/Pipeline.png)

### Environment

This source code was tested in the following environment:

Python = 3.8.18  
PyTorch = 1.11.0  
torchvision = 0.12.0  
Ubuntu 18.04.5 LTS  
NVIDIA RTX A6000

### Dataset

* (1) Download the datasets and organize the structure as follows:
```
dataset folder
├── train
│   ├── class_001
│   │      ├── 1.jpg
│   │      ├── 2.jpg
│   │      └── ...
│   ├── class_002
│   │      ├── 1.jpg
│   │      ├── 2.jpg
│   │      └── ...
│   └── ...
├── validation
│   ├── class_001
│   │      ├── 1.jpg
│   │      ├── 2.jpg
│   │      └── ...
│   ├── class_002
│   │      ├── 1.jpg
│   │      ├── 2.jpg
│   │      └── ...
│   └── ...
└── test
    ├── class_001
    │      ├── 1.jpg
    │      ├── 2.jpg
    │      └── ...
    ├── class_002
    │      ├── 1.jpg
    │      ├── 2.jpg
    │      └── ...
    └── ...
```

* (2) modify the path to the dataset folders.

### Dependencies

* PyTorch_Sobel
```
cd ECT-BoFM
git clone https://github.com/zhaoyuzhi/PyTorch-Sobel.git
```
Note: Rename the `PyTorch-Sobel` folder to `PyTorch_Sobel`, and rename the `pytorch-sobel.py` file inside it to `pytorch_sobel.py`.

* Bagnet
```
pip install git+https://github.com/wielandbrendel/bag-of-local-features-models.git
```

### Training and Inference
```
python train_inference.py
```

### Visualization Results
![enter image description here](https://github.com/Dichao-Liu/ECT-BoFM/blob/main/visualization.png)

---

### Citation

If you find this work useful, please cite the following paper:

```bibtex
@Article{diagnostics15151883,
  AUTHOR = {Liu, Dichao and Suzuki, Kenji},
  TITLE = {Edge- and Color–Texture-Aware Bag-of-Local-Features Model for Accurate and Interpretable Skin Lesion Diagnosis},
  JOURNAL = {Diagnostics},
  VOLUME = {15},
  YEAR = {2025},
  NUMBER = {15},
  ARTICLE-NUMBER = {1883},
  URL = {https://www.mdpi.com/2075-4418/15/15/1883},
  ISSN = {2075-4418},
  DOI = {10.3390/diagnostics15151883}
}
```
