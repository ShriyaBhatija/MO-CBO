import numpy as np

def is_pareto_optimal(point_cloud):
    is_optimal= np.ones(point_cloud.shape[0], dtype = bool)
    for i, c in enumerate(point_cloud):
        if is_optimal[i]:
            is_optimal[is_optimal] = np.any(point_cloud[is_optimal]<c, axis=1)  
            is_optimal[i] = True  
    return is_optimal

def generational_distance(A, Z, p=2):
    A = np.array(A)
    Z = np.array(Z)
    distances = []
    for a in A:
        min_distance = np.min(np.linalg.norm(Z - a, ord=p, axis=1))
        distances.append(min_distance ** p)  
    return ((np.sum(distances)) / len(A))**(1/p)