import autograd as ag
import click
import copy
import numpy as np
import logging
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils import check_random_state

from recnn.preprocessing import permute_by_pt
from recnn.preprocessing import extract
from recnn.recnn import log_loss
from recnn.recnn import adam
from recnn.recnn import grnn_init_simple_join
from recnn.recnn import grnn_predict_simple_join


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(levelname)s] %(message)s")


@click.command()
@click.argument("filename_train1")
@click.argument("filename_train2")
@click.argument("filename_model")
@click.option("--n_features", default=7)
@click.option("--n_hidden", default=40)
@click.option("--n_epochs", default=20)
@click.option("--batch_size", default=64)
@click.option("--step_size", default=0.0005)
@click.option("--decay", default=0.9)
@click.option("--random_state", default=1)
def train(filename_train1,
          filename_train2,
          filename_model,
          n_features=7,
          n_hidden=40,
          n_epochs=5,
          batch_size=64,
          step_size=0.0005,
          decay=0.9,
          random_state=1):
    # Initialization
    logging.info("Calling with...")
    logging.info("\tfilename_train1 = %s" % filename_train1)
    logging.info("\tfilename_train2 = %s" % filename_train2)
    logging.info("\tfilename_model = %s" % filename_model)
    logging.info("\tn_features = %d" % n_features)
    logging.info("\tn_hidden = %d" % n_hidden)
    logging.info("\tn_epochs = %d" % n_epochs)
    logging.info("\tbatch_size = %d" % batch_size)
    logging.info("\tstep_size = %f" % step_size)
    logging.info("\tdecay = %f" % decay)
    logging.info("\trandom_state = %d" % random_state)
    rng = check_random_state(random_state)

    # Make data
    logging.info("Loading data...")

    fd = open(filename_train1, "rb")
    X1, y = pickle.load(fd)
    fd.close()
    y = np.array(y)

    fd = open(filename_train2, "rb")
    X2, _ = pickle.load(fd)
    fd.close()

    indices = rng.permutation(len(X1))
    size = min(80000, len(X1))
    X1 = [X1[i] for i in indices[:size]]
    X2 = [X2[i] for i in indices[:size]]
    y = [y[i] for i in indices[:size]]
    y = np.array(y)

    logging.info("\tfilename = %s" % filename_train1)
    logging.info("\tfilename = %s" % filename_train2)
    logging.info("\tX1 size = %d" % len(X1))
    logging.info("\tX2 size = %d" % len(X2))
    logging.info("\ty size = %d" % len(y))

    # Preprocessing
    logging.info("Preprocessing...")

    X1 = [extract(permute_by_pt(jet)) for jet in X1]
    tf = RobustScaler().fit(np.vstack([jet["content"] for jet in X1]))

    for jet in X1:
        jet["content"] = tf.transform(jet["content"])

    X2 = [extract(permute_by_pt(jet)) for jet in X2]
    tf = RobustScaler().fit(np.vstack([jet["content"] for jet in X2]))

    for jet in X2:
        jet["content"] = tf.transform(jet["content"])

    # Split into train+test
    logging.info("Splitting into train and validation...")

    X1_train, X1_valid, X2_train, X2_valid, y_train, y_valid = train_test_split(X1, X2, y, test_size=5000, random_state=rng)

    # Training
    logging.info("Training...")

    predict = grnn_predict_simple_join
    init = grnn_init_simple_join

    trained_params = init(n_features, n_hidden, random_state=rng)
    n_batches = int(np.ceil(len(X1_train) / batch_size))
    best_score = [-np.inf]  # yuck, but works
    best_params = [trained_params]

    def loss(X1, X2, y, params):
        y_pred = predict(params, X1, X2)
        l = log_loss(y, y_pred).mean()
        return l

    def objective(params, iteration):
        rng = check_random_state(iteration % n_batches)
        start = rng.randint(len(X1_train) - batch_size)
        idx = slice(start, start+batch_size)
        return loss(X1_train[idx], X2_train[idx], y_train[idx], params)

    def callback(params, iteration, gradient):
        if iteration % 25 == 0:
            roc_auc = roc_auc_score(y_valid,
                                    predict(params, X1_valid, X2_valid))

            if roc_auc > best_score[0]:
                best_score[0] = roc_auc
                best_params[0] = copy.deepcopy(params)

                fd = open(filename_model, "wb")
                pickle.dump(best_params[0], fd)
                fd.close()

            logging.info(
                "%5d\t~loss(train)=%.4f\tloss(valid)=%.4f"
                "\troc_auc(valid)=%.4f\tbest_roc_auc(valid)=%.4f" % (
                    iteration,
                    loss(X1_train[:5000], X2_train[:5000],
                         y_train[:5000], params),
                    loss(X1_valid, X2_valid,
                         y_valid, params),
                    roc_auc,
                    best_score[0]))

    for i in range(n_epochs):
        logging.info("epoch = %d" % i)
        logging.info("step_size = %.4f" % step_size)

        trained_params = adam(ag.grad(objective),
                              trained_params,
                              step_size=step_size,
                              num_iters=1 * n_batches,
                              callback=callback)
        step_size = step_size * decay


if __name__ == "__main__":
    train()
