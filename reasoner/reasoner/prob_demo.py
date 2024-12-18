"""
Simple demo of how the end probability of target object is calculated.

"""
import numpy as np

# Data for demo
demo_objects = [
    {
        "name":"011_banana_1",
        "color": "yellow",
        "confidence": 0.9,
        "position": [3,3,1]
    },
    {
        "name":"010_apple_1",
        "color": "red",
        "confidence": 0.9,
        "position": [4,8,1]
    },
    {
        "name":"011_banana_2",
        "color": "yellow",
        "confidence": 0.9,
        "position": [9,7,1]
    },
    {
        "name":"011_banana_3",
        "color": "yellow",
        "confidence": 0.9,
        "position": [9,3,1]
    },
    {
        "name":"010_apple_2",
        "color": "red",
        "confidence": 0.9,
        "position": [11,3,1]
    },
]
max_x = 14
max_y = 10

# Functions for demo
def calculate_distances(objects: list, point: list):
    # returns list of distances of objects to point
    distances = []
    for obj in objects:
        position = obj["position"]
        distances.append(np.linalg.norm(np.array(point) - np.array(position)))
    return distances

def evaluate_distances(distances: list, sigma=0.4):
    # returns list of probabilities from distances
    # distances to Gaussian distribution and normalized
    unnormalized_p = []
    for dist in distances:
        prob = np.exp(-(dist**2) / (2 * sigma**2))
        unnormalized_p.append(prob)
    normalized_p = unnormalized_p / np.sum(unnormalized_p)
    return normalized_p

def evaluate_reference(objects: list, ref: int):
    # return list of normalized sum of distance probabilities
    # distances of object to target object
    position = objects[ref]["position"]
    name = objects[ref]["name"]

    distances_from_target = []
    for i in range(len(objects)):
        obj = objects[i]
        # print(obj["name"][:3], name[:3], obj["name"][:3] == name[:3])
        if obj["name"][:3] == name[:3]:
            d = np.linalg.norm(np.array(position)-np.array(obj["position"]))
            # print(d)
            distances_from_target.append(d)
        else:
            distances_from_target.append(0.0)
    dist_sum = np.sum(distances_from_target)
    for j in range(len(distances_from_target)):
        value = distances_from_target[j]
        if value == 0.0:
            distances_from_target[j] = 0.0
        elif value == dist_sum:
            distances_from_target[j] = 1.0
        else:
            # normalized and inversed (closer -> bigger)
            distances_from_target[j] = (dist_sum - value) / dist_sum
    return distances_from_target
        
def evaluate_objects(objects: list, vector: list):
    # return output probabilities from human vector
    distances_from_vector = calculate_distances(objects, vector)
    dist_prob = evaluate_distances(distances_from_vector,sigma=2.0)

    rows = []
    for idx in range(len(objects)):
        row = evaluate_reference(objects,idx)
        rows.append(list(dist_prob[idx]*objects[idx]["confidence"]*np.array(row)))
        # rows.append(row)
    
    prob_matrix = np.array(rows)
    # print(prob_matrix)
    ret = np.sum(prob_matrix,axis=0)
    norm = np.sum(ret)
    return ret / norm

def print_array(array):
    # print float array in readable way
    cnt = 0
    for number in array:
        cnt += 1
        if cnt <= len(array)-1:
            print("%.3f" % number,end=", ")
        else:
            print("%.3f" % number,end="")
    print()

# Demo for probability calculations
rnd_vector = np.random.rand(2)
# human_vector = [int(rnd_vector[0]*max_x), int(rnd_vector[1]*max_y), 1]
human_vector = [10,5,1]
print(f"Human is poiting at {human_vector[:2]}")

distances_from_human = calculate_distances(demo_objects, human_vector)
# print_array(distances_from_human)
dist_probs = evaluate_distances(distances_from_human,sigma=2.0)
# print_array(dist_probs)

out_probs = evaluate_objects(demo_objects, human_vector)
print_array(out_probs)
max_out = np.argmax(out_probs)
print()
print(demo_objects[max_out]["name"])