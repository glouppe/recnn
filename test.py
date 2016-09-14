import numpy as np
import logging
import pickle

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils import check_random_state

from recursive_jet.preprocessing import permute_by_pt
from recursive_jet.preprocessing import rotate
from recursive_jet.recnn import grnn_predict_gated

rng = check_random_state(1)
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(levelname)s] %(message)s")


# Make training data ----------------------------------------------------------
logging.info("Loading training data...")

filename = "data/w-vs-qcd/kt-train.pickle"
fd = open(filename, "rb")
X, y = pickle.load(fd)
fd.close()
y = np.array(y)

logging.info("\tfilename = %s" % filename)
logging.info("\tX size = %d" % len(X))
logging.info("\ty size = %d" % len(y))


# Preprocessing ---------------------------------------------------------------
logging.info("Preprocessing...")
X = [rotate(permute_by_pt(jet)) for jet in X]
tf = RobustScaler().fit(np.vstack([jet["content"] for jet in X]))

for jet in X:
    jet["content"] = tf.transform(jet["content"])


# Make test data -------------------------------------------------------------
logging.info("Loading test data...")

filename = "data/w-vs-qcd/kt-test.pickle"
fd = open(filename, "rb")
X, y = pickle.load(fd)
fd.close()
y = np.array(y)

logging.info("\tfilename = %s" % filename)
logging.info("\tX size = %d" % len(X))
logging.info("\ty size = %d" % len(y))


# Preprocessing ---------------------------------------------------------------
logging.info("Preprocessing...")
X = [rotate(permute_by_pt(jet)) for jet in X]

for jet in X:
    jet["content"] = tf.transform(jet["content"])


# Loading model ---------------------------------------------------------------
logging.info("Loading model...")

fd = open("models/w-vs-qcd-params.pickle", "rb")
params = pickle.load(fd)
fd.close()


# Testing ---------------------------------------------------------------
logging.info("Testing...")

predict = grnn_predict_gated
logging.info("roc_auc(test)=%.4f" % roc_auc_score(y, predict(params, X)))
