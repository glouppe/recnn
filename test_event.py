import click
import numpy as np
import logging
import pickle

from sklearn.preprocessing import RobustScaler
from sklearn.utils import check_random_state

from recnn.preprocessing import rewrite_content
from recnn.preprocessing import permute_by_pt
from recnn.preprocessing import extract
from recnn.recnn import event_init
from recnn.recnn import event_predict


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(levelname)s] %(message)s")


@click.command()
@click.argument("filename_train")
@click.argument("filename_test")
@click.argument("filename_model")
@click.argument("n_events_train")
@click.argument("n_events_test")
@click.argument("filename_output")
@click.option("--n_jets_per_event", default=10)
@click.option("--random_state", default=1)
def test(filename_train,
         filename_test,
         filename_model,
         n_events_train,
         n_events_test,
         filename_output,
         n_jets_per_event=10,
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
    logging.info("\tn_jets_per_event = %d" % n_jets_per_event)
    logging.info("\trandom_state = %d" % random_state)
    rng = check_random_state(random_state)

    # Make data
    logging.info("Loading train data + preprocessing...")

    fd = open(filename_train, "rb")

    # training file is assumed to be formatted a sequence of pickled pairs
    # (e_i, y_i), where e_i is a list of (phi, eta, pt, mass, jet) tuples.

    X = []
    y = []

    for i in range(n_events_train):
        e_i, y_i = pickle.load(fd)

        original_features = []
        jets = []

        for j, (phi, eta, pt, mass, jet) in enumerate(e_i[:n_jets_per_event]):
            if len(jet["tree"]) > 1:
                original_features.append((phi, eta, pt, mass))
                jet = extract(permute_by_pt(rewrite_content(jet)))
                jets.append(jet)

        if len(jets) == n_jets_per_event:
            X.append([np.array(original_features), jets])
            y.append(y_i)

    y = np.array(y)

    fd.close()

    logging.info("\tfilename = %s" % filename_train)
    logging.info("\tX size = %d" % len(X))
    logging.info("\ty size = %d" % len(y))

    # Building scalers
    logging.info("Building scalers...")
    tf_features = RobustScaler().fit(
        np.vstack([features for features, _ in X]))
    tf_content = RobustScaler().fit(
        np.vstack([j["content"] for _, jets in X for j in jets]))

    X = None
    y = None

    # Loading test data
    logging.info("Loading test data + preprocessing...")

    fd = open(filename_test, "rb")

    # training file is assumed to be formatted a sequence of pickled pairs
    # (e_i, y_i), where e_i is a list of (phi, eta, pt, mass, jet) tuples.

    X = []
    y = []

    for i in range(n_events_test):
        e_i, y_i = pickle.load(fd)

        original_features = []
        jets = []

        for j, (phi, eta, pt, mass, jet) in enumerate(e_i[:n_jets_per_event]):
            if len(jet["tree"]) > 1:
                original_features.append((phi, eta, pt, mass))
                jet = extract(permute_by_pt(rewrite_content(jet)))
                jets.append(jet)

        if len(jets) == n_jets_per_event:
            X.append([np.array(original_features), jets])
            y.append(y_i)

    y = np.array(y)

    fd.close()

    logging.info("\tfilename = %s" % filename_train)
    logging.info("\tX size = %d" % len(X))
    logging.info("\ty size = %d" % len(y))

    # Scaling
    logging.info("Scaling...")

    for i in range(len(X)):
        X[i][0] = tf_features.transform(X[i][0])

        for j in X[i][1]:
            j["content"] = tf_content.transform(j["content"])

    # Testing
    logging.info("Testing...")

    predict = event_predict

    fd = open(filename_model, "rb")
    params = pickle.load(fd)
    fd.close()

    all_y_pred = []

    for start in range(0, len(y), 1000):
        y_pred = predict(params, X[start:start+1000],
                         n_jets_per_event=n_jets_per_event)
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
