#!/usr/bin/python3

import rclpy, zmq, re, json
import numpy as np

from reasoner.msg import DeicticSolution, HRICommand, GDRNSolution
from std_msgs.msg import Header
from classes import PointingInput, LanguageInput, GDRNetInput, ReasonerTester
from rclpy.node import Node

class Reasoner(Node):
  """
  Poiting, Language and GDRNet merger node
  """
  def __init__(self):
    super().__init__('tiago_reasoner_node')
    self.relations = ["color", "shape", "left", "right"]
    
    # Initialize subscriber classes
    self.pointing_sub = PointingInput(self)
    self.language_sub = LanguageInput(self)
    self.gdrnet_sub = GDRNetInput(self)
    
    # Initialize publisher
    self.pub = self.create_publisher(HRICommand, 'tiago_hri_output', 10)
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.PUB)
    self.socket.bind("tcp://*:5558")
    
    # Create timer for main processing loop
    self.timer = self.create_timer(2.0, self.main_loop)
    
    self.get_logger().info('Tiago Merger node initialized')
        
  def main_loop(self):
    # Main processing loop
    self.action = self.language_sub.data["action"]
    self.action_param = self.language_sub.data["action_param"]
    self.lang_objects = self.language_sub.data["object"]
    self.lang_objects_param = self.language_sub["object_param"]
    
    self.deitic = self.pointing_sub.get_solution()
    self.gdrn_objects = self.gdrnet_sub.get_objects()
    
    if self.action_param in self.relations:
      # if known action param and if target objects
      if self.nlp_empty():
        self.target_probs = self.eval_pointing_ref()
      else:
        self.target_probs = self.eval_language_ref()
        pass
      
    else:
      if self.nlp_empty():
        # pick object human is pointing at
        self.target_probs = self.eval_pointing()
      else:
        self.target_probs = self.eval_language()
    
    self.publish_results()

  def wait(self, duration:int):
    # simple wait duration number of rates
    self.get_logger().info(f"Waiting for {duration} seconds...")
    rate = self.create_rate(1.0)
    for _ in range(duration):
      rate.sleep()
    self.get_logger().info("Done waiting!")

  def get_gdrn_object_names(self):
    """
    Returns list of all object names is GDRN objects
    """
    names = []
    for obj in self.gdrn_objects:
      names.append(obj["name"])
    return names

  def nlp_empty(self):
    """
    Check if returned nlp objects list is empty or have null or 
    nonsensical objects inside
    """
    ret = True
    gdrn_names = self.get_gdrn_object_names()
    for target in self.lang_objects:
      if type(target) != str:
        break
      for name in gdrn_names:
        name = name[4:] # strip gdrnet numbers
        name = re.match(r'^([^_]*)', name).group(1) # strip gdrnet index number
        # all the above makes from 011_banana_1, banana
        if target == name:
          ret = False
      
    return ret

  def meet_ref_criteria(self, ref: dict, obj: dict, tolerance=0.1):
    # return True if given object meets filter criteria of reference
    ret = False
    if (self.action_param == "color" and obj["color"] == ref["color"]):
      ret = True
    elif (self.action_param == "shape" and obj["name"][:3] == ref["color"][:3]):
      ret - True
    elif self.action_param in ["left", "right"]:
      dir_vector = [
        self.deitic.line_point_1[0] - self.deitic.line_point_2[0], 
        self.deitic.line_point_1[1] - self.deitic.line_point_2[1]
        ]
      ref_pos = ref["position"]
      obj_pos = obj["position"]
      
      ref_to_pos = [ref_pos[0] - obj_pos[0], ref_pos[1] - obj_pos[1]]
      dot_product = (
        ref_to_pos[0] * -dir_vector[1] + 
        ref_to_pos[1] * dir_vector[0]
        )
      if self.action_param == "left" and dot_product < -tolerance:
        ret = True
      elif self.action_param == "right" and dot_product > tolerance:
        ret = True
    
    return ret

  def eval_pointing_ref(self):
    """
    Calculates probabilities of target (GDRNet++) objects with reference
    from pointing input
    """
    # demo_prob.py implementation
    
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
      # return list of probabilities for related objects of reference
      ref_obj = objects[ref]

      target_probs = []
      # firstly we fill probs list with distances for related objects
      for i in range(len(objects)):
        obj = objects[i]
        # print(obj["name"][:3], name[:3], obj["name"][:3] == name[:3])
        if self.meet_ref_criteria(obj,ref_obj):
          d = np.linalg.norm(
            np.array(ref_obj["position"]) - np.array(obj["position"])
            )
          # print(d)
          target_probs.append(d)
        else:
          # for later sum unrelated objects needs to equal zero
          target_probs.append(0.0)
      
      # compute prob from distances
      dist_sum = np.sum(target_probs)
      for j in range(len(target_probs)):
        value = target_probs[j]
        if value == 0.0:
            target_probs[j] = 0.0
        elif value == dist_sum:
            target_probs[j] = 1.0
        else:
            # normalized and inversed (closer -> bigger)
            target_probs[j] = (dist_sum - value) / dist_sum
      return target_probs

    dist_probs = evaluate_distances(self.deitic.distances_from_line,sigma=2.0)
    
    rows = []
    for i in range(len(self.gdrn_objects)):
      row = evaluate_reference(self.gdrn_objects,i)
      rows.append(list(
        dist_probs[i] * self.gdrn_objects[i]["confidence"] * np.array(row)
        ))
    
    probs_matrix = np.array(rows)
    ret = np.sum(probs_matrix,axis=0)
    return ret / np.sum(ret)
  
  def eval_pointing(self):
    """
    Calculates probabilty of target (GDRNet++) objects by their
    distance to pointing input
    """ 
    
    def evaluate_distances(distances: list, sigma=0.4):
      # returns list of probabilities from distances
      # distances to Gaussian distribution and normalized
      unnormalized_p = []
      for dist in distances:
          prob = np.exp(-(dist**2) / (2 * sigma**2))
          unnormalized_p.append(prob)
      normalized_p = unnormalized_p / np.sum(unnormalized_p)
      return normalized_p
    
    def get_confidence_scores(objects: list):
      # return list of confidence score of given gdrn objects list
      scores = []
      for obj in objects:
        scores.append(obj["confidence"])
      return scores
    
    probs = evaluate_distances(self.deitic.distances_from_line, sigma=2.0)
    probs = np.array(probs) * np.array(get_confidence_scores(self.gdrn_objects))
    
    return probs
  
  def eval_language_ref(self):
    """
    Calculates probabilities of target (GDRNet++) objects with reference
    from language input
    """
    
    def find_indexes(objects: list, param: str):
      # returns list of object's indexes from language
      ret = []
      for i in range(len(objects)):
        obj_name = objects[i]["name"][4:] # strip gdrnet type number
        obj_name = re.match(r'^([^_]*)', obj_name).group(1) # strip index number
        if obj_name == param:
          ret.append(i)
      return ret
    
    def evaluate_reference(objects: list, ref: int):
      # return list of probabilities for related objects of reference
      ref_obj = objects[ref]

      target_probs = []
      # firstly we fill probs list with distances for related objects
      for i in range(len(objects)):
        obj = objects[i]
        if self.meet_ref_criteria(obj,ref_obj):
          d = np.linalg.norm(
            np.array(ref_obj["position"]) - np.array(obj["position"])
            )
          # print(d)
          target_probs.append(d)
        else:
          # for later sum unrelated objects needs to equal zero
          target_probs.append(0.0)
      
      # compute prob from distances
      dist_sum = np.sum(target_probs)
      for j in range(len(target_probs)):
        value = target_probs[j]
        if value == 0.0:
            target_probs[j] = 0.0
        elif value == dist_sum:
            target_probs[j] = 1.0
        else:
            # normalized and inversed (closer -> bigger)
            target_probs[j] = (dist_sum - value) / dist_sum
      return target_probs
    target = self.lang_objects[0] # first target object from language
    print(target)
    indexes = find_indexes(self.lang_objects, target)
    prob = 1 / len(indexes)
    
    rows = []
    for i in indexes:
      row = evaluate_reference(self.gdrn_objects,i)
      rows.append(list(
        prob * self.gdrn_objects[i]["confidence"] * np.array(row)
        ))
    
    probs_matrix = np.array(rows)
    ret = np.sum(probs_matrix,axis=0)
    return ret / np.sum(ret)
  
  def eval_language(self):
    """
    Calculates probabilty of target (GDRNet++) objects by their
    """
    def find_objects(objects: list, param: str):
      # returns list of object's indexes from language
      ret = []
      for obj in objects:
        obj_name = obj["name"][4:] # strip gdrnet type number
        obj_name = re.match(r'^([^_]*)', obj_name).group(1) # strip index number
        if obj_name == param:
          ret.append(obj)
      return ret
    
    def evaluate_objects(objects: list, sigma=2.0):
      # returns prob list based on object distance to robot
      unnormalized = []
      for obj in objects:
        dist = np.linalg.norm(obj["position"])
        prob = np.exp(-(dist**2) / (2 * sigma**2))
        unnormalized.append(prob)
      ret = unnormalized / np.sum(unnormalized)
      return ret
    
    language_objects = find_objects(self.gdrn_objects, self.lang_objects[0])
    probs = evaluate_objects(language_objects,sigma=2.0)
    return probs

  def publish_results(self):
    """
    Creates HRI command and publishes it to ros2 topic and ZeroMQ (for noetic)
    """
    data_dict = {
      "action": ["pick"],
      "action_probs": [1.0],
      "objects": self.get_gdrn_object_names(),
      "object_probs": self.target_probs 
    }
    
    # Create HRICommand
    msg = HRICommand()
    msg.header = Header()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = "zmq_publisher"

    msg.data = json.dumps(data_dict)
    
    serialized_msg = json.dumps({
        "header": {
            "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
            "frame_id": msg.header.frame_id
        },
        "data": msg.data
    })
    self.socket.send_string(serialized_msg)
    self.pub.publish(msg)


if __name__ == '__main__':
  rclpy.init()
  
  # Create and spin the main node
  merger_node = Reasoner()
  # Testing class, change test: str ("color", "shape", "left" and "right")
  tester = ReasonerTester(merger_node,"shape")
  
  try:
    merger_node.wait(5)
    rclpy.spin_once(merger_node)
  except KeyboardInterrupt:
    pass
  finally:
    merger_node.destroy_node()
    rclpy.shutdown()
    
