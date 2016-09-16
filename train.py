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
from recnn.preprocessing import rotate
from recnn.recnn import log_loss
from recnn.recnn import adam
from recnn.recnn import grnn_init_simple
from recnn.recnn import grnn_predict_simple
from recnn.recnn import grnn_init_gated
from recnn.recnn import grnn_predict_gated


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(levelname)s] %(message)s")


@click.command()
@click.argument("filename_train")
@click.argument("filename_model")
@click.option("--simple", is_flag=True, default=False)
@click.option("--n_features", default=7)
@click.option("--n_hidden", default=30)
@click.option("--n_epochs", default=5)
@click.option("--batch_size", default=64)
@click.option("--step_size", default=0.01)
@click.option("--decay", default=0.7)
@click.option("--random_state", default=1)
def train(filename_train,
          filename_model,
          simple=False,
          n_features=7,
          n_hidden=30,
          n_epochs=5,
          batch_size=64,
          step_size=0.01,
          decay=0.7,
          random_state=1):
    # Initialization
    gated = not simple
    logging.info("Calling with...")
    logging.info("\tfilename_train = %s" % filename_train)
    logging.info("\tfilename_model = %s" % filename_model)
    logging.info("\tgated = %s" % gated)
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

    fd = open(filename_train, "rb")
    X, y = pickle.load(fd)
    fd.close()
    y = np.array(y)

    logging.info("\tfilename = %s" % filename_train)
    logging.info("\tX size = %d" % len(X))
    logging.info("\ty size = %d" % len(y))

    # Preprocessing
    logging.info("Preprocessing...")
    X = [rotate(permute_by_pt(jet)) for jet in X]
    tf = RobustScaler().fit(np.vstack([jet["content"] for jet in X]))

    for jet in X:
        jet["content"] = tf.transform(jet["content"])

    # Split into train+test
    logging.info("Splitting into train and validation...")

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=5000,
                                                          random_state=rng)

    # Training
    logging.info("Training...")

    if gated:
        predict = grnn_predict_gated
        init = grnn_init_gated
    else:
        predict = grnn_predict_simple
        init = grnn_init_simple

    trained_params = init(n_features, n_hidden, random_state=rng)
    n_batches = int(np.ceil(len(X_train) / batch_size))
    best_score = [-np.inf]  # yuck, but works
    best_params = [trained_params]

    def loss(X, y, params):
        y_pred = predict(params, X)
        l = log_loss(y, y_pred).mean()
        return l

    def objective(params, iteration):
        rng = check_random_state(iteration % n_batches)
        start = rng.randint(len(X_train) - batch_size)
        idx = slice(start, start+batch_size)
        return loss(X_train[idx], y_train[idx], params)

    def callback(params, iteration, gradient):
        if iteration % 25 == 0:
            roc_auc = roc_auc_score(y_valid, predict(params, X_valid))

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
                    loss(X_train[:5000], y_train[:5000], params),
                    loss(X_valid, y_valid, params),
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
