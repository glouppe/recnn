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

from recnn.recnn import log_loss
from recnn.recnn import adam
from recnn.recnn import event_baseline_init
from recnn.recnn import event_baseline_predict


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(levelname)s] %(message)s")


@click.command()
@click.argument("filename_train")
@click.argument("filename_model")
@click.argument("n_events")
@click.option("--n_features_rnn", default=4)
@click.option("--n_hidden_rnn", default=40)
@click.option("--n_epochs", default=20)
@click.option("--batch_size", default=64)
@click.option("--step_size", default=0.0005)
@click.option("--decay", default=0.9)
@click.option("--n_particles_per_event", default=10)
@click.option("--random_state", default=1)
def train(filename_train,
          filename_model,
          n_events,
          n_features_rnn=4,
          n_hidden_rnn=40,
          n_epochs=5,
          batch_size=64,
          step_size=0.01,
          decay=0.7,
          n_particles_per_event=10,
          random_state=1):
    # Initialization
    n_events = int(n_events)
    logging.info("Calling with...")
    logging.info("\tfilename_train = %s" % filename_train)
    logging.info("\tfilename_model = %s" % filename_model)
    logging.info("\tn_events = %d" % n_events)
    logging.info("\tn_features_rnn = %d" % n_features_rnn)
    logging.info("\tn_hidden_rnn = %d" % n_hidden_rnn)
    logging.info("\tn_epochs = %d" % n_epochs)
    logging.info("\tbatch_size = %d" % batch_size)
    logging.info("\tstep_size = %f" % step_size)
    logging.info("\tdecay = %f" % decay)
    logging.info("\tn_particles_per_event = %d" % n_particles_per_event)
    logging.info("\trandom_state = %d" % random_state)
    rng = check_random_state(random_state)

    # Make data
    logging.info("Loading data + preprocessing...")

    fd = open(filename_train, "rb")

    X = []
    y = []

    for i in range(n_events):
        v_i, y_i = pickle.load(fd)
        v_i = v_i[:n_particles_per_event]

        X.append(v_i)
        y.append(y_i)

    y = np.array(y)

    fd.close()

    logging.info("\tfilename = %s" % filename_train)
    logging.info("\tX size = %d" % len(X))
    logging.info("\ty size = %d" % len(y))

    # Preprocessing
    logging.info("Preprocessing...")
    tf_features = RobustScaler().fit(
        np.vstack([features for features in X]))

    for i in range(len(X)):
        X[i] = tf_features.transform(X[i])

        if len(X[i]) < n_particles_per_event:
            X[i] = np.vstack([X[i],
                              np.zeros((n_particles_per_event - len(X[i]), 4))])

    # Split into train+test
    logging.info("Splitting into train and validation...")

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=1000,
                                                          stratify=y,
                                                          random_state=rng)

    # Training
    logging.info("Training...")

    predict = event_baseline_predict
    init = event_baseline_init

    trained_params = init(n_features_rnn, n_hidden_rnn,
                          random_state=rng)

    n_batches = int(np.ceil(len(X_train) / batch_size))
    best_score = [-np.inf]  # yuck, but works
    best_params = [trained_params]

    def loss(X, y, params):
        y_pred = predict(params, X,
                         n_particles_per_event=n_particles_per_event)
        l = log_loss(y, y_pred).mean()
        return l

    def objective(params, iteration):
        rng = check_random_state(iteration)
        start = rng.randint(len(X_train) - batch_size)
        idx = slice(start, start+batch_size)
        return loss(X_train[idx], y_train[idx], params)

    def callback(params, iteration, gradient):
        if iteration % 25 == 0:
            roc_auc = roc_auc_score(y_valid,
                                    predict(params, X_valid,
                                            n_particles_per_event=n_particles_per_event))

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
