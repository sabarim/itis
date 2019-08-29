# Iteratively Trained Interactive Segmentation

## This is an official TensorFlow implementation for

[
Sabarinath Mahadevan, Paul Voigtlaender, and Bastian Leibe,  
"Iteratively Trained Interactive Segmentation",  
British Machine Vision Conference, 2018.
](http://bmvc2018.org/contents/papers/0652.pdf)

## Citation

If you use this code or models, please cite the following:

```bibtex
@inproceedings{mahadevanitis,
  author={Sabarinath Mahadevan and Paul Voigtlaender and Bastian Leibe},
  title={Iteratively Trained Interactive Segmentation},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2018},
}
```

## Pre-trained Models

You can download the pre-trained models from our [internal server](https://omnomnom.vision.rwth-aachen.de/data/itis/).
All available models are in a single tar.gz file. Currently it contains models that can be used to reproduce the results for iFCN and ITIS in Table 1, and for the ablation study in Figure 5 (see [paper](http://bmvc2018.org/contents/papers/0652.pdf) for details.


## Usage

* Download PascalVOC dataset (http://host.robots.ox.ac.uk/pascal/VOC/) 
* Create a folder data within the source root directory, and copy the Pascal VOC dataset files to it. Alternatively, add a parameter "data_dir: \<path to pascal voc root\>" in the respective config files.
* Download the weights as explained in the previous section, and place them under 'models' directory. Alternatively, you could change the paramter "load" to point it to the required path.
* Run the following to evaluate the given models
  
```misc
iFCN:         python main.py configs/pascal_ifcn
iFCN + gauss: python main.py configs/pascal_gauss
ITIS:         python main.py configs/pascal_itis
```

