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
        "position": [1.1,-0.3,0.6]
    },
    {
        "name":"010_apple_1",
        "color": "red",
        "confidence": 0.9,
        "position": [1.4,0.2,0.6]
    },
    {
        "name":"011_banana_2",
        "color": "yellow",
        "confidence": 0.9,
        "position": [1.05,0.1,0.6]
    },
    {
        "name":"006_lemon_1",
        "color": "yellow",
        "confidence": 0.9,
        "position": [1.5,-0.2,0.6]
    },
    {
        "name":"003_tomato_sauce_1",
        "color": "red",
        "confidence": 0.9,
        "position": [1.2,0.0,0.6544]
    },
]

# Demo for probability calculations
human_vector = [1.2,-0.3,0.63] # banana 1
action_param = "left"
print(f"Human is poiting at {human_vector[:2]} and action param: {action_param}")


# Functions for demo
def meet_ref_criteria(ref: dict, obj: dict, tolerance=0.01):
    # return True if given object meets filter criteria of reference
    def check_shape(ref: dict, obj: dict) -> bool:
        ref_num = int(ref["name"][:3])
        obj_num = int(obj["name"][:3])
        return ref_num == obj_num
    
    def check_color(ref: dict, obj: dict) -> bool:
        ret = False
        if type(ref["color"]) == str and type(obj["color"]) == str:
            ret = (ref["color"] == obj["color"])
        elif type(ref["color"]) == list and type(obj["color"]) == str:
            ret = (obj["color"] in ref["color"])
        elif type(ref["color"]) == str and type(obj["color"]) == list:
            ret = (ref["color"] in obj["color"])
        elif type(ref["color"]) == list and type(obj["color"]) == list:
            for ref_color in ref["color"]:
                if ref_color in obj["color"]:
                    ret = True
        return ret
    
    ret = False
    if (action_param == "color" and check_color(ref,obj)):
        ret = True
    elif (action_param == "shape" and check_shape(ref,obj)):
        ret = True
    elif action_param in ["left", "right"]:
        dir_vector = human_vector[:2]
        # print(f"Pointing vector: {dir_vector}")
        ref_pos = ref["position"]
        obj_pos = obj["position"]
      
        ref_to_pos = [obj_pos[0] - ref_pos[0], obj_pos[1] - ref_pos[1]]
        dot_product = (
          ref_to_pos[0] * -dir_vector[1] + 
          ref_to_pos[1] * dir_vector[0] )

        # print(f"{obj['name'] }, {obj['position'] },{dot_product}")
        if action_param == "right" and dot_product < -tolerance:
            ret = True
        elif action_param == "left" and dot_product > tolerance:
            ret = True
    
    # print(f"return: {ret}, ref, obj: {ref_num}, {obj_num}")
    return ret

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

def evaluate_reference(objects: list, ref_idx: int):
        # return list of probabilities for related objects of reference object
        ref = objects[ref_idx]
        dist_to_ref = []
        # firstly we calculate 1/distances to reference object
        for i in range(len(objects)):
            obj = objects[i]
            # print(ref_obj["name"],obj["name"])
            if meet_ref_criteria(ref,obj):
                dist = np.linalg.norm(
                    np.array(obj["position"]) - np.array(ref["position"]) )
                dist_to_ref.append(1/dist)
            else:
                # for later sum unrelated objects needs to equal zero
                dist_to_ref.append(0.0)
        
        # compute prob from distances
        ref_probs = []
        if np.sum(dist_to_ref) != 0.0:
            ref_probs = np.array(dist_to_ref) / np.sum(dist_to_ref)
        else:
            ref_probs = np.zeros(len(objects))
        
        print(ref["name"])
        print_array(dist_to_ref)
        return list(ref_probs)
        
def evaluate_objects(objects: list, vector: list):
    # return output probabilities from human vector
    distances_from_vector = calculate_distances(objects, vector)
    print("distances from human vector")
    print_array(distances_from_vector)
    dist_prob = evaluate_distances(distances_from_vector,sigma=0.2)
    print_array(dist_prob)
    print("---")

    rows = []
    for idx in range(len(objects)):
        row = evaluate_reference(objects,idx)
        print_array(row)
        print("-")
        rows.append(list(dist_prob[idx]*objects[idx]["confidence"]*np.array(row)))
        # rows.append(row)
    
    prob_matrix = np.array(rows)
    # print_matrix(prob_matrix)
    ret = np.sum(prob_matrix,axis=0)
    # print_array(ret)
    # print()
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

def print_matrix(matrix):
    for i in range(len(matrix)):
        print("[",end="")
        for j in range(len(matrix[0])):
            if j == len(matrix[0]) - 1:
                print(round(matrix[i][j],4), end="")
            else:
                print(round(matrix[i][j],4),end=", ")
        print("]",end="\n")
    
#=================
out_probs = evaluate_objects(demo_objects, human_vector)

print("\n--- out")
print_array(out_probs)
for idx in range(len(demo_objects)):
    if idx == len(demo_objects)-1:
        print(demo_objects[idx]["name"][4:])
    else:
        print(demo_objects[idx]["name"][4:], end=" ")
print()
max_out = np.argmax(out_probs)
print()
print(demo_objects[max_out]["name"])