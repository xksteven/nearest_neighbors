import numpy as np

from numpy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm


def normalize(x):
    return x / norm(x)

def compute_batch_dist(mat, mat2, dist="l2", num_neighbors=1):
    #print(mat.shape, mat2.shape)
    tmp_mat = mat.reshape((1, -1))
    tmp_mat2 = mat2.reshape((mat2.shape[0], -1))
    #print(mat.shape, mat2.shape)
    result = euclidean_distances(tmp_mat, tmp_mat2)
    sorted_indices = result.argsort()[-num_neighbors:][::-1]
    #print(result, result.shape)
    return result[0, sorted_indices][0], sorted_indices

def compute_dist(vector, vector2, dist="l2"):
    # L2 distance
    if dist == "l2":
        result = norm(vector - vector2)
    elif dist == "cosine":
        vector2 = normalize(vector2)
        result = 1 - norm_vector.dot(vector2)
    else:
        print("distance fn not implemented")
        exit()
    return result

def add_to_nn_array(indices, num_neighbors, nn_index, index2, result, min_value, orig_value=None, sec_value=None):
    if len(indices[-1]) < num_neighbors:
        nn_index = index2
        min_value = result
        if orig_value is not None:
            indices[-1].append([nn_index, min_value, orig_value, sec_value])
        else:
            indices[-1].append([nn_index, min_value, ])
        # TODO convert to a heap
        indices[-1] = sorted(indices[-1], key=lambda x: x[1])
        min_value = indices[-1][-1][1]  # the largest of the values
    elif result < min_value:
        nn_index = index2
        min_value = result
        if orig_value is not None:
            indices[-1].append([nn_index, min_value, orig_value, sec_value])
        else:
            indices[-1].append([nn_index, min_value, ])
        indices[-1] = sorted(indices[-1], key=lambda x: x[1])
        del indices[-1][-1]  # remove the largest
        min_value = indices[-1][-1][1]  # the largest of the values
    return nn_index, min_value, indices

def compute_nn(x1, x2, y1=None, y2=None, same_label=False, any_label=True, 
               ignore_same_index=False, num_neighbors=1, progress=False, dist="l2"):
    """
    Find nearest neighbors of x1 in x2
    Returns
        indices of nearest neighbors of x1 in x2.
    """
    indices = []
    if len(x1) == 0 or len(x2) == 0:
        print("array inputs 1 or 2 is empty.")
        return [[]]

    if progress:
        pbar = tqdm(total=len(x1))
    for index, vector in enumerate(x1):
        nn_index = 0
        min_value = np.inf
        norm_vector = normalize(vector)
        indices.append([])
        for index2, data in enumerate(x2):
            if ignore_same_index and (index == index2):
                continue
            if any_label:
                pass
            elif same_label and not (np.array_equal(y1[index], y2[index2])):
                continue
            elif not same_label and (np.array_equal(y1[index], y2[index2])):
                continue


            results, sorted_indices = compute_batch_dist(vector, data, dist=dist, num_neighbors=num_neighbors)
            # if passing in a matrix for data use the below
            # TODO need to check n_dim
            data = data[sorted_indices]
            for index3, result in enumerate(results):
                nn_index, min_value, indices = add_to_nn_array(
                    indices, num_neighbors,
                    nn_index, index2, result, min_value, orig_value=vector, sec_value=data[index])

        #indices.append([nn_index, min_value])
        if progress:
            pbar.update(1)
    if progress:
        pbar.close()
    return np.array(indices)
