import click
import numpy as np
import logging
import pickle

from sklearn.preprocessing import RobustScaler
from sklearn.utils import check_random_state

from recnn.recnn import event_baseline_predict


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(levelname)s] %(message)s")


@click.command()
@click.argument("filename_train")
@click.argument("filename_test")
@click.argument("filename_model")
@click.argument("n_events_train")
@click.argument("n_events_test")
@click.argument("filename_output")
@click.option("--n_particles_per_event", default=10)
@click.option("--random_state", default=1)
def test(filename_train,
         filename_test,
         filename_model,
         n_events_train,
         n_events_test,
         filename_output,
         n_particles_per_event=10,
         random_state=1):
    # Initialization
    n_events_train = int(n_events_train)
    n_events_test = int(n_events_test)
    logging.info("Calling with...")
    logging.info("\tfilename_train = %s" % filename_train)
    logging.info("\tfilename_test = %s" % filename_test)
    logging.info("\tfilename_model = %s" % filename_model)
    logging.info("\tn_events_train = %d" % n_events_train)
    logging.info("\tn_events_test = %d" % n_events_test)
    logging.info("\tfilename_output = %s" % filename_output)
    logging.info("\tn_particles_per_event = %d" % n_particles_per_event)
    logging.info("\trandom_state = %d" % random_state)
    rng = check_random_state(random_state)

    # Make data
    logging.info("Loading train data + preprocessing...")

    fd = open(filename_train, "rb")

    X = []
    y = []

    for i in range(n_events_train):
        v_i, y_i = pickle.load(fd)
        v_i = v_i[:n_particles_per_event]

        X.append(v_i)
        y.append(y_i)

    y = np.array(y)

    fd.close()

    logging.info("\tfilename = %s" % filename_train)
    logging.info("\tX size = %d" % len(X))
    logging.info("\ty size = %d" % len(y))

    # Building scalers
    logging.info("Building scalers...")
    tf_features = RobustScaler().fit(
        np.vstack([features for features in X]))

    X = None
    y = None

    # Loading test data
    logging.info("Loading test data + preprocessing...")

    fd = open(filename_test, "rb")

    X = []
    y = []

    for i in range(n_events_test):
        v_i, y_i = pickle.load(fd)
        v_i = v_i[:n_particles_per_event]

        X.append(v_i)
        y.append(y_i)

    y = np.array(y)

    fd.close()

    logging.info("\tfilename = %s" % filename_train)
    logging.info("\tX size = %d" % len(X))
    logging.info("\ty size = %d" % len(y))

    # Scaling
    logging.info("Scaling...")

    for i in range(len(X)):
        X[i] = tf_features.transform(X[i])

        if len(X[i]) < n_particles_per_event:
            X[i] = np.vstack([X[i],
                              np.zeros((n_particles_per_event - len(X[i]), 4))])

    # Testing
    logging.info("Testing...")

    predict = event_baseline_predict

    fd = open(filename_model, "rb")
    params = pickle.load(fd)
    fd.close()

    all_y_pred = []

    for start in range(0, len(y), 1000):
        y_pred = predict(params, X[start:start+1000],
                         n_particles_per_event=n_particles_per_event)
        all_y_pred.append(y_pred)

    y_pred = np.concatenate(all_y_pred)

    # Save
    output = np.hstack((y.reshape(-1, 1),
                        y_pred.reshape(-1, 1)))

    fd = open(filename_output, "wb")
    pickle.dump(output, fd, protocol=2)
    fd.close()


if __name__ == "__main__":
    test()
