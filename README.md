# QCD-Aware Recursive Neural Networks for Jet Physics
https://arxiv.org/abs/1702.00748

* Gilles Louppe
* Kyunghyun Cho
* Cyril Becot
* Kyle Cranmer

Recent progress in applying machine learning for jet physics has been built upon an analogy between calorimeters and images. In this work, we present a novel class of recursive neural networks built instead upon an analogy between QCD and natural languages. In the analogy, four-momenta are like words and the clustering history of sequential recombination jet algorithms is like the parsing of a sentence. Our approach works directly with the four-momenta of a variable-length set of particles, and the jet-based tree structure varies on an event-by-event basis. Our experiments highlight the flexibility of our method for building task-specific jet embeddings and show that recursive architectures are significantly more accurate and data efficient than previous image-based networks. We extend the analogy from individual jets (sentences) to full events (paragraphs), and show for the first time an event-level classifier operating on all the stable particles produced in an LHC event.

---

Please cite using the following BibTex entry:

```
@article{louppe2017qcdrecnn,
           author = {{Louppe}, G. and {Cho}, K. and {Becot}, C and {Cranmer}, K.},
            title = "{QCD-Aware Recursive Neural Networks for Jet Physics}",
          journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
           eprint = {1702.00748},
     primaryClass = "hep-th",
             year = 2017,
            month = feb,
}
```

---

## Instructions

### Requirements

- python 2.7
- [autograd](https://github.com/HIPS/autograd/tree/master/autograd)
- scikit-learn
- click

### Data

(mirror to be released)

### Usage

Classification of W vs QCD jets:

```
# Training
python train.py data/w-vs-qcd/final/antikt-kt-train.pickle model.pickle

# Test
# see notebooks/04-jet-study.ipynb
```

Classification of full events:

```
# Training
python train_event.py data/events/antikt-kt-train.pickle model.pickle

# Test
python test_event.py data/events/antikt-kt-train.pickle data/events/antikt-kt-test.pickle model.pickle 100000 100000 predictions.pickle
# see also notebooks/04-event-study.ipynb
```

### Rebuilding the data

Additional requirements:

- [fastjet](http://fastjet.fr/)
- cython
- rootpy
- h5py

Building W vs QCD jet data:
1. Load data from h5 files. Execute cells 1-3 from `notebooks/01-load-w-vs-qcd.ipynb`. Adapt cell 3 as necessary.
2. Split the data into train and test. Execute cells 1-3 from `notebooks/02-split-train-test.ipynb`. Adapt cell 3 as necessary.
3. Preprocess the data. Execute cells 1-6 from `notebooks/03-preprocessing.ipynb`.
Adapt cell 6 as necessary. Run one of the following cells depending on the desired topology.
4. (Optional) Repack the train and test using the last cell of `notebooks/03-preprocessing.ipynb`. This is necessary for historical reasons to exactly reproduce the paper data.

Building full event data:
1. Load data from pickle files. Execute cells 1-2, 4-7. Adapt cell 7 as necessary.
2. Split the data into train and test. Execute cells 1-2 and the last of `notebooks/02-split-train-test.ipynb`.
