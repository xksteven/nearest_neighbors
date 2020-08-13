# nearest_neighbors
Compute nearest neighbors given numpy arrays or a pytorch model

# Usage

Find nearest neighbors of x1 in x2

    indices, values = compute_nn(x1, x2) 
    # also accepts labels for x1, and x2, namely y1, and y2
    # can optionally filter the inputs based on whether you want y1 == y2
    
    
