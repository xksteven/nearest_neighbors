import numpy as np

from tqdm import tqdm
from numpy.linalg import norm


def normalize(x):
    return x / norm(x)

def compute_nn(x1, x2, y1=None, y2=None, same_label=False, any_label=True, 
               ignore_same_index=False, nn=1, progress=False, dist="l2"):
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
        for index2, vector2 in enumerate(x2):
            if ignore_same_index and (index == index2):
                continue
            if any_label:
                pass
            elif same_label and not (np.array_equal(y1[index], y2[index2])):
                continue
            elif not same_label and (np.array_equal(y1[index], y2[index2])):
                continue
            #vector2 = normalize(vector2)
            # dot product for normalized vectors
            #result = vector.dot(vector2)
            #if result > min_value:

            # L2 distance
            if dist == "l2":
                result = norm(vector - vector2)
            elif dist == "cosine":
                vector2 = normalize(vector2) 
                result = 1 - norm_vector.dot(vector2)
            else:
                print("distance fn not implemented")
                exit()
            
            if len(indices[-1]) < nn:
                nn_index = index2
                min_value = result
                indices[-1].append([nn_index, min_value])
                indices[-1] = sorted(indices[-1], key=lambda x: x[1])
                min_value = indices[-1][-1][1]  # the largest of the values
            elif result < min_value:
                nn_index = index2
                min_value = result
                indices[-1].append([nn_index, min_value])
                indices[-1] = sorted(indices[-1], key=lambda x: x[1])
                #print(len(indices), indices[-1][-1])
                del indices[-1][-1]  # remove the largest
                min_value = indices[-1][-1][1]  # the largest of the values

        #indices.append([nn_index, min_value])
        if progress:
            pbar.update(1)
    if progress:
        pbar.close()
    return np.array(indices)
