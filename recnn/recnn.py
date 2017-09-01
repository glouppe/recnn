import autograd.numpy as np
from autograd.util import flatten_func
from sklearn.utils import check_random_state


# Batchization of the recursion

def batch(jets):
    # Batch the recursive activations across all nodes of a same level
    # !!! Assume that jets have at least one inner node.
    #     Leads to off-by-one errors otherwise :(

    # Reindex node IDs over all jets
    #
    # jet_children: array of shape [n_nodes, 2]
    #     jet_children[node_id, 0] is the node_id of the left child of node_id
    #     jet_children[node_id, 1] is the node_id of the right child of node_id
    #
    # jet_contents: array of shape [n_nodes, n_features]
    #     jet_contents[node_id] is the feature vector of node_id
    jet_children = []
    offset = 0

    for jet in jets:
        tree = np.copy(jet["tree"])
        tree[tree != -1] += offset
        jet_children.append(tree)
        offset += len(tree)

    jet_children = np.vstack(jet_children)
    jet_contents = np.vstack([jet["content"] for jet in jets])
    n_nodes = offset

    # Level-wise traversal
    level_children = np.zeros((n_nodes, 4), dtype=np.int32)
    level_children[:, [0, 2]] -= 1

    inners = []   # Inner nodes at level i
    outers = []   # Outer nodes at level i
    offset = 0

    for jet in jets:
        queue = [(jet["root_id"] + offset, -1, True, 0)]

        while len(queue) > 0:
            node, parent, is_left, depth = queue.pop(0)

            if len(inners) < depth + 1:
                inners.append([])
            if len(outers) < depth + 1:
                outers.append([])

            # Inner node
            if jet_children[node, 0] != -1:
                inners[depth].append(node)
                position = len(inners[depth]) - 1
                is_leaf = False

                queue.append((jet_children[node, 0], node, True, depth + 1))
                queue.append((jet_children[node, 1], node, False, depth + 1))

            # Outer node
            else:
                outers[depth].append(node)
                position = len(outers[depth]) - 1
                is_leaf = True

            # Register node at its parent
            if parent >= 0:
                if is_left:
                    level_children[parent, 0] = position
                    level_children[parent, 1] = is_leaf
                else:
                    level_children[parent, 2] = position
                    level_children[parent, 3] = is_leaf

        offset += len(jet["tree"])

    # Reorganize levels[i] so that inner nodes appear first, then outer nodes
    levels = []
    n_inners = []
    contents = []

    prev_inner = np.array([], dtype=int)

    for inner, outer in zip(inners, outers):
        n_inners.append(len(inner))
        inner = np.array(inner, dtype=int)
        outer = np.array(outer, dtype=int)
        levels.append(np.concatenate((inner, outer)))

        left = prev_inner[level_children[prev_inner, 1] == 1]
        level_children[left, 0] += len(inner)
        right = prev_inner[level_children[prev_inner, 3] == 1]
        level_children[right, 2] += len(inner)

        contents.append(jet_contents[levels[-1]])

        prev_inner = inner

    # levels: list of arrays
    #     levels[i][j] is a node id at a level i in one of the trees
    #     inner nodes are positioned within levels[i][:n_inners[i]], while
    #     leaves are positioned within levels[i][n_inners[i]:]
    #
    # level_children: array of shape [n_nodes, 2]
    #     level_children[node_id, 0] is the position j in the next level of
    #         the left child of node_id
    #     level_children[node_id, 1] is the position j in the next level of
    #         the right child of node_id
    #
    # n_inners: list of shape len(levels)
    #     n_inners[i] is the number of inner nodes at level i, accross all
    #     trees
    #
    # contents: array of shape [n_nodes, n_features]
    #     contents[sum(len(l) for l in layers[:i]) + j] is the feature vector
    #     or node layers[i][j]

    return (levels, level_children[:, [0, 2]], n_inners, contents)


# Activations

def sigmoid(x):
    return 0.5 * (np.tanh(x) + 1.0)


def relu(x, alpha=0.0):
    if alpha == 0.0:
        return 0.5 * (x + np.abs(x))
    else:
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        return f1 * x + f2 * np.abs(x)


def logsumexp(X):
    max_X = np.max(X, axis=-1)[..., np.newaxis]
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=-1)[..., np.newaxis])


def softmax(X):
    return np.exp(X - logsumexp(X))


# Initializations

def glorot_uniform(fan_in, fan_out, rng, scale=0.1):
    s = scale * np.sqrt(6. / (fan_in + fan_out))
    if fan_out > 0:
        return rng.rand(fan_in, fan_out) * 2 * s - s
    else:
        return rng.rand(fan_in) * 2 * s - s


def orthogonal(shape, rng, scale=1.1):
    # from Keras
    a = rng.normal(0.0, 1.0, shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == shape else v
    q = q.reshape(shape)
    return scale * q


# Simple recursive activation

def grnn_init_simple(n_features, n_hidden, random_state=None):
    rng = check_random_state(random_state)
    return {"W_u": glorot_uniform(n_hidden, n_features, rng),
            "b_u": np.zeros(n_hidden),
            "W_h": orthogonal((n_hidden, 3 * n_hidden), rng),
            "b_h": np.zeros(n_hidden),
            "W_clf": [glorot_uniform(n_hidden, n_hidden, rng),
                      glorot_uniform(n_hidden, n_hidden, rng),
                      glorot_uniform(n_hidden, 0, rng)],
            "b_clf": [np.zeros(n_hidden),
                      np.zeros(n_hidden),
                      np.ones(1)]}


def grnn_transform_simple(params, jets):
    levels, children, n_inners, contents = batch(jets)
    n_levels = len(levels)
    embeddings = []

    for i, nodes in enumerate(levels[::-1]):
        j = n_levels - 1 - i
        inner = nodes[:n_inners[j]]
        outer = nodes[n_inners[j]:]

        u_k = relu(np.dot(params["W_u"], contents[j].T).T + params["b_u"])

        if len(inner) > 0:
            h_L = embeddings[-1][children[inner, 0]]
            h_R = embeddings[-1][children[inner, 1]]
            h = relu(np.dot(params["W_h"],
                            np.hstack((h_L, h_R, u_k[:n_inners[j]])).T).T +
                     params["b_h"])

            embeddings.append(np.concatenate((h, u_k[n_inners[j]:])))

        else:
            embeddings.append(u_k)

    return embeddings[-1].reshape((len(jets), -1))


def grnn_predict_simple(params, jets):
    h = grnn_transform_simple(params, jets)

    h = relu(np.dot(params["W_clf"][0], h.T).T + params["b_clf"][0])
    h = relu(np.dot(params["W_clf"][1], h.T).T + params["b_clf"][1])
    h = sigmoid(np.dot(params["W_clf"][2], h.T).T + params["b_clf"][2])

    return h.ravel()


# Gated recursive activation

def grnn_init_gated(n_features, n_hidden, random_state=None):
    rng = check_random_state(random_state)

    return {"W_u": glorot_uniform(n_hidden, n_features, rng),
            "b_u": np.zeros(n_hidden),
            "W_h": orthogonal((n_hidden, 3 * n_hidden), rng),
            "b_h": np.zeros(n_hidden),
            "W_z": glorot_uniform(4 * n_hidden, 4 * n_hidden, rng),
            "b_z": np.zeros(4 * n_hidden),
            "W_r": glorot_uniform(3 * n_hidden, 3 * n_hidden, rng),
            "b_r": np.zeros(3 * n_hidden),
            "W_clf": [glorot_uniform(n_hidden, n_hidden, rng),
                      glorot_uniform(n_hidden, n_hidden, rng),
                      glorot_uniform(n_hidden, 0, rng)],
            "b_clf": [np.zeros(n_hidden),
                      np.zeros(n_hidden),
                      np.ones(1)]}


def grnn_transform_gated(params, jets, return_states=False):
    levels, children, n_inners, contents = batch(jets)
    n_levels = len(levels)
    n_hidden = len(params["b_u"])

    if return_states:
        states = {"embeddings": [], "z": [], "r": [], "levels": levels,
                  "children": children, "n_inners": n_inners}

    embeddings = []

    for i, nodes in enumerate(levels[::-1]):
        j = n_levels - 1 - i
        inner = nodes[:n_inners[j]]
        outer = nodes[n_inners[j]:]

        u_k = relu(np.dot(params["W_u"], contents[j].T).T + params["b_u"])

        if len(inner) > 0:
            u_k_inners = u_k[:n_inners[j]]
            u_k_leaves = u_k[n_inners[j]:]

            h_L = embeddings[-1][children[inner, 0]]
            h_R = embeddings[-1][children[inner, 1]]

            hhu = np.hstack((h_L, h_R, u_k_inners))
            r = sigmoid(np.dot(params["W_r"], hhu.T).T + params["b_r"])
            h_H = relu(np.dot(params["W_h"], np.multiply(r, hhu).T).T +
                       params["b_h"])

            z = np.dot(params["W_z"],
                       np.hstack((h_H, hhu)).T).T + params["b_z"]
            z_H = z[:, :n_hidden]               # new activation
            z_L = z[:, n_hidden:2*n_hidden]     # left activation
            z_R = z[:, 2*n_hidden:3*n_hidden]   # right activation
            z_N = z[:, 3*n_hidden:]             # local state
            z = np.concatenate([z_H[..., np.newaxis],
                                z_L[..., np.newaxis],
                                z_R[..., np.newaxis],
                                z_N[..., np.newaxis]], axis=2)
            z = softmax(z)

            h = (np.multiply(z[:, :, 0], h_H) +
                 np.multiply(z[:, :, 1], h_L) +
                 np.multiply(z[:, :, 2], h_R) +
                 np.multiply(z[:, :, 3], u_k_inners))

            embeddings.append(np.vstack((h, u_k_leaves)))

            if return_states:
                states["embeddings"].append(embeddings[-1])
                states["z"].append(z)
                states["r"].append(r)

        else:
            embeddings.append(u_k)

            if return_states:
                states["embeddings"].append(embeddings[-1])

    if return_states:
        return states
    else:
        return embeddings[-1].reshape((len(jets), -1))


def grnn_predict_gated(params, jets):
    h = grnn_transform_gated(params, jets)

    h = relu(np.dot(params["W_clf"][0], h.T).T + params["b_clf"][0])
    h = relu(np.dot(params["W_clf"][1], h.T).T + params["b_clf"][1])
    h = sigmoid(np.dot(params["W_clf"][2], h.T).T + params["b_clf"][2])

    return h.ravel()


# Event-level classification
def event_init(n_features_embedding,
               n_hidden_embedding,
               n_features_rnn,
               n_hidden_rnn,
               n_jets_per_event,
               random_state=None):
    rng = check_random_state(random_state)
    params = grnn_init_simple(n_features_embedding,
                              n_hidden_embedding,
                              random_state=rng)

    params.update({
        "rnn_W_hh": orthogonal((n_hidden_rnn, n_hidden_rnn), rng),
        "rnn_W_hx": glorot_uniform(n_hidden_rnn, n_features_rnn, rng),
        "rnn_b_h": np.zeros(n_hidden_rnn),
        "rnn_W_zh": orthogonal((n_hidden_rnn, n_hidden_rnn,), rng),
        "rnn_W_zx": glorot_uniform(n_hidden_rnn, n_features_rnn, rng),
        "rnn_b_z": np.zeros(n_hidden_rnn),
        "rnn_W_rh": orthogonal((n_hidden_rnn, n_hidden_rnn,), rng),
        "rnn_W_rx": glorot_uniform(n_hidden_rnn, n_features_rnn, rng),
        "rnn_b_r": np.zeros(n_hidden_rnn),
        "W_clf": [glorot_uniform(n_hidden_rnn, n_hidden_rnn, rng),
                  glorot_uniform(n_hidden_rnn, n_hidden_rnn, rng),
                  glorot_uniform(n_hidden_rnn, 0, rng)],
        "b_clf": [np.zeros(n_hidden_rnn),
                  np.zeros(n_hidden_rnn),
                  np.ones(1)]
        })

    return params


def event_transform(params, X, n_jets_per_event=10):
    # Assume events e_j are structured as pairs (features, jets)
    # where features is a N_j x n_features array
    #       jets is a list of N_j jets

    # Convert jets
    jets = []
    features = []

    for e in X:
        features.append(e[0][:n_jets_per_event])
        jets.extend(e[1][:n_jets_per_event])

    h_jets = np.hstack([
        np.vstack(features),
        grnn_transform_simple(params, jets)])
    h_jets = h_jets.reshape(len(X), n_jets_per_event, -1)

    # GRU layer
    h = np.zeros((len(X), params["rnn_b_h"].shape[0]))

    for t in range(n_jets_per_event):
        xt = h_jets[:, n_jets_per_event - 1 - t, :]
        zt = sigmoid(np.dot(params["rnn_W_zh"], h.T).T +
                     np.dot(params["rnn_W_zx"], xt.T).T + params["rnn_b_z"])
        rt = sigmoid(np.dot(params["rnn_W_rh"], h.T).T +
                     np.dot(params["rnn_W_rx"], xt.T).T + params["rnn_b_r"])
        ht = relu(np.dot(params["rnn_W_hh"], np.multiply(rt, h).T).T +
                  np.dot(params["rnn_W_hx"], xt.T).T + params["rnn_b_h"])
        h = np.multiply(1. - zt, h) + np.multiply(zt, ht)

    return h


def event_predict(params, X, n_jets_per_event=10):
    h = event_transform(params, X,
                        n_jets_per_event=n_jets_per_event)

    h = relu(np.dot(params["W_clf"][0], h.T).T + params["b_clf"][0])
    h = relu(np.dot(params["W_clf"][1], h.T).T + params["b_clf"][1])
    h = sigmoid(np.dot(params["W_clf"][2], h.T).T + params["b_clf"][2])

    return h.ravel()


# Event baseline (direct gru)
def event_baseline_init(n_features_rnn,
                        n_hidden_rnn,
                        random_state=None):
    rng = check_random_state(random_state)
    params = {}

    params.update({
        "rnn_W_hh": orthogonal((n_hidden_rnn, n_hidden_rnn), rng),
        "rnn_W_hx": glorot_uniform(n_hidden_rnn, n_features_rnn, rng),
        "rnn_b_h": np.zeros(n_hidden_rnn),
        "rnn_W_zh": orthogonal((n_hidden_rnn, n_hidden_rnn,), rng),
        "rnn_W_zx": glorot_uniform(n_hidden_rnn, n_features_rnn, rng),
        "rnn_b_z": np.zeros(n_hidden_rnn),
        "rnn_W_rh": orthogonal((n_hidden_rnn, n_hidden_rnn,), rng),
        "rnn_W_rx": glorot_uniform(n_hidden_rnn, n_features_rnn, rng),
        "rnn_b_r": np.zeros(n_hidden_rnn),
        "W_clf": [glorot_uniform(n_hidden_rnn, n_hidden_rnn, rng),
                  glorot_uniform(n_hidden_rnn, n_hidden_rnn, rng),
                  glorot_uniform(n_hidden_rnn, 0, rng)],
        "b_clf": [np.zeros(n_hidden_rnn),
                  np.zeros(n_hidden_rnn),
                  np.ones(1)]
        })

    return params


def event_baseline_transform(params, X, n_particles_per_event=10):
    features = []

    for e in X:
        features.append(e[:n_particles_per_event])

    h_jets = np.vstack(features)
    h_jets = h_jets.reshape(len(X), n_particles_per_event, -1)

    # GRU layer
    h = np.zeros((len(X), params["rnn_b_h"].shape[0]))

    for t in range(n_particles_per_event):
        xt = h_jets[:, n_particles_per_event - 1 - t, :]
        zt = sigmoid(np.dot(params["rnn_W_zh"], h.T).T +
                     np.dot(params["rnn_W_zx"], xt.T).T + params["rnn_b_z"])
        rt = sigmoid(np.dot(params["rnn_W_rh"], h.T).T +
                     np.dot(params["rnn_W_rx"], xt.T).T + params["rnn_b_r"])
        ht = relu(np.dot(params["rnn_W_hh"], np.multiply(rt, h).T).T +
                  np.dot(params["rnn_W_hx"], xt.T).T + params["rnn_b_h"])
        h = np.multiply(1. - zt, h) + np.multiply(zt, ht)

    return h


def event_baseline_predict(params, X, n_particles_per_event=10):
    h = event_baseline_transform(params, X,
                                 n_particles_per_event=n_particles_per_event)

    h = relu(np.dot(params["W_clf"][0], h.T).T + params["b_clf"][0])
    h = relu(np.dot(params["W_clf"][1], h.T).T + params["b_clf"][1])
    h = sigmoid(np.dot(params["W_clf"][2], h.T).T + params["b_clf"][2])

    return h.ravel()


# Training

def log_loss(y, y_pred):
    return -(y * np.log(y_pred) + (1. - y) * np.log(1. - y_pred))


def square_error(y, y_pred):
    return (y - y_pred) ** 2


def sgd(grad, init_params,
        callback=None, num_iters=200, step_size=0.1, mass=0.9):
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    velocity = np.zeros(len(x))

    for i in range(num_iters):
        g = flattened_grad(x, i)

        if callback:
            callback(unflatten(x), i, unflatten(g))

        velocity = mass * velocity - (1.0 - mass) * g
        x = x + step_size * velocity

    return unflatten(x)


def rmsprop(grad, init_params, callback=None, num_iters=100,
            step_size=0.1, gamma=0.9, eps=10**-8):
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    avg_sq_grad = np.ones(len(x))

    for i in range(num_iters):
        g = flattened_grad(x, i)

        if callback:
            callback(unflatten(x), i, unflatten(g))

        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x = x - step_size * g/(np.sqrt(avg_sq_grad) + eps)

    return unflatten(x)


def adam(grad, init_params, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    m = np.zeros(len(x))
    v = np.zeros(len(x))

    for i in range(num_iters):
        g = flattened_grad(x, i)

        if callback:
            callback(unflatten(x), i, unflatten(g))

        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)

    return unflatten(x)
