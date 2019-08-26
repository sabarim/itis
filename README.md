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

```misc
iFCN:         python main.py pascal_ifcn
iFCN + gauss: python main.py pascal_gauss
ITIS:         python main.py pascal_itis
```
