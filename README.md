# SeaweedSight: Estimating biomass density in land-based cultivation of Ulva spp. using a low cost RGB imaging system
[[`paper`](google.com)]
[[`dataset`](https://drive.google.com/drive/folders/1UBy1mnVWgmaF_1nKgGyrYtS_L52mACIy?usp=sharing)]


> The code associated with [['paper]](google.com) where we developed a low-cost RGB imaging system for *Ulva spp*. biomass estimation in land-based raceways. Using segmentation and power regression models, we achieve accurate biomass density predictions to potentially reduce labor costs and achieve more routine monitoring. 

<img src="./doc/Ulva_05_1_cycle3_example.gif" width="1000">

From the example above of *Ulva spp*. footage recorded at a biomass density of 0.5 g/L and processed using the [Segment Anything Model](segmentanything.com). While model mispredictions can be seen, these are smoothed and removed through aggregating frame-level data into per-revolution levels, leading to more robust predictors and estimates.

## Monitoring setup
An IDS UI-5290-FA-C-HQ camera with a 7 mm lens was mounted approximately 80 centimeters above the raceway (see Figure), with a field of view of 75 cm so that the full width of the raceway was in view of the camera. For more details, please see the manuscript.

<img src="./doc/camera_setup.jpg" width="1000">

## Setup

```
# install seaweedsight and its dependencies
pip install git+ssh://git@github.com/geoJoost/SeaweedSight.git

# Setup the environment
conda create -n seaweedsight
conda activate seaweedsight

conda install pip

# Modify this installation link to the correct CUDA/CPU version
# See: https://pytorch.org/get-started/locally/
pip3 install torch torchvision

pip install --upgrade huggingface_hub transformers

conda install -c conda-forge numpy pandas scikit-learn scikit-image opencv matplotlib seaborn statsmodels pillow

```
## Getting started
To reproduce the results in the manuscript:
1. Download the dataset from [Zenodo](google.com) and place it in `/data/` folder at the root of this project.
2. Run the script: `main.py` with default parameters

Testing on your own dataset:
1. Organize your dataset in the same structures as the 
* Run `main.py` without modifications

To test on your own dataset:
1. Record footage of your cultivation system in similar circumstances to the GIF above.
2. Store the .mp4 or .avi in `/data/inference/`
3. Run the script `inference.py` **NOT IMPLEMENTED YET**

---
If you use this code or dataset, please cite our [paper](google.com). For questions, feedback, or collaborations, feel free to [contact us](mailto:joost.vandalen@wur.nl).

<img src="./doc/Ulva_05_1_cycle3_example.gif" width="1000">
