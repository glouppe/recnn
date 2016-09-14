import numpy as np
import logging
import pickle
import sys

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils import check_random_state

from recnn.preprocessing import permute_by_pt
from recnn.preprocessing import rotate
from recnn.recnn import grnn_predict_gated


if len(sys.argv) != 4:
    print("Usage: python test.py train-data.pickle test-data.pickle model.pickle")
    sys.exit(1)
else:
    filename_train = sys.argv[1]
    filename_test = sys.argv[2]
    filename_model = sys.argv[3]


rng = check_random_state(1)
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(levelname)s] %(message)s")


# Make training data ----------------------------------------------------------
logging.info("Loading training data...")

fd = open(filename_train, "rb")
X, y = pickle.load(fd)
fd.close()
y = np.array(y)

logging.info("\tfilename = %s" % filename_train)
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

fd = open(filename_test, "rb")
X, y = pickle.load(fd)
fd.close()
y = np.array(y)

logging.info("\tfilename = %s" % filename_test)
logging.info("\tX size = %d" % len(X))
logging.info("\ty size = %d" % len(y))


# Preprocessing ---------------------------------------------------------------
logging.info("Preprocessing...")
X = [rotate(permute_by_pt(jet)) for jet in X]

for jet in X:
    jet["content"] = tf.transform(jet["content"])


# Loading model ---------------------------------------------------------------
logging.info("Loading model...")

fd = open(filename_model, "rb")
params = pickle.load(fd)
fd.close()

logging.info("\tfilename = %s" % filename_model)


# Testing ---------------------------------------------------------------
logging.info("Testing...")

predict = grnn_predict_gated
logging.info("roc_auc(test)=%.4f" % roc_auc_score(y, predict(params, X)))
