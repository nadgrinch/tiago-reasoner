#!/usr/bin/python3

import rclpy, zmq, re, json, time
import numpy as np

from threading import Lock
from gesture_msgs.msg import HRICommand, DeicticSolution
from scene_msgs.msg import GDRNObject, GDRNSolution
from std_msgs.msg import Header
from reasoner.classes import PointingInput, LanguageInput, GDRNetInput, ReasonerTester
from rclpy.node import Node

class Reasoner(Node):
  """
  Poiting, Language and GDRNet merger node
  """
  def __init__(self, sigma=0.4):
    super().__init__('tiago_reasoner_node')
    self.relations = ["color", "shape", "left", "right"]
    self.sigma = sigma
    
    # Initialize subscriber classes
    self.pointing_sub = PointingInput(self)
    self.language_sub = LanguageInput(self)
    self.gdrnet_sub = GDRNetInput(self)
    
    # Initialize publisher
    self.pub = self.create_publisher(HRICommand, 'tiago_hri_output', 10)
    context = zmq.Context()
    self.socket = context.socket(zmq.PUB)
    self.socket.bind("tcp://*:5558")
    
    # Create timer for main processing loop
    self.timer = self.create_timer(5.0, self.main_loop)
    
    self.get_logger().info('Tiago Merger node initialized')
        
  def main_loop(self):
    # Main processing loop
    if self.language_sub.data == None:
      self.get_logger().warn("No NLP input, waiting")
      return
    
    self.action = self.language_sub.data["action"]
    self.action_param = self.language_sub.data["action_param"][0]
    self.lang_objects = self.language_sub.data["objects"]
    self.lang_objects_param = self.language_sub.data["object_param"]
    self.language_sub.data = None

    self.deitic = self.pointing_sub.get_solution()
    self.gdrn_objects = self.gdrnet_sub.get_objects()
    
    self.get_logger().info(f"Action parameter: {self.action_param}")
    if self.action_param in self.relations:
      # if known action param and if target objects
      if self.nlp_empty():
        self.get_logger().info(f"Find target of gesture with action parameter")
        self.target_probs = self.eval_pointing_param()
      else:
        self.get_logger().info(f"Find target of NLP with action parameter")
        self.target_probs = self.eval_language_param()
        pass
      
    else:
      if self.nlp_empty():
        # pick object human is pointing at
        self.get_logger().info("Find target of gesture")
        self.target_probs = self.eval_pointing()
      else:
        self.get_logger().info("Find target of NLP")
        self.target_probs = self.eval_language()
    
    self.publish_results()

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
        stripped = re.match(r'^([^_]*)', name[4:]).group(1) # strip gdrnet index number
        # all the above makes from 011_banana_1, banana
        if target == stripped:
          ret = False
      
    return ret

  def meet_ref_criteria(self, ref: dict, obj: dict, tolerance=0.01):
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
    if (self.action_param == "color" and check_color(ref,obj)):
      ret = True
    elif (self.action_param == "shape" and check_shape(ref,obj)):
      ret = True
    elif self.action_param in ["left", "right"]:
      dir_vector = [
        self.deitic.line_point_1.x - self.deitic.line_point_2.x, 
        self.deitic.line_point_1.y - self.deitic.line_point_2.y ]
      # print(f"Pointing vector: {dir_vector}")
      ref_pos = ref["position"]
      obj_pos = obj["position"]
      
      ref_to_pos = [obj_pos[0] - ref_pos[0], obj_pos[1] - ref_pos[1]]
      dot_product = (
        ref_to_pos[0] * -dir_vector[1] + 
        ref_to_pos[1] * dir_vector[0] )

      # print(f"{obj['name'] }, {obj['position'] },{dot_product}")
      if self.action_param == "right" and dot_product < -tolerance:
        ret = True
      elif self.action_param == "left" and dot_product > tolerance:
        ret = True
    
    # print(f"return: {ret}, ref, obj: {ref_num}, {obj_num}")
    return ret

  def evaluate_distances(self, distances: list, sigma=0.4):
      # returns list of probabilities from distances
      # distances to Gaussian distribution and normalized
      unnormalized_p = []
      for dist in distances:
          prob = np.exp(-(dist**2) / (2 * sigma**2))
          unnormalized_p.append(prob)
      normalized_p = unnormalized_p / np.sum(unnormalized_p)
      return list(normalized_p)
  
  def evaluate_reference(self, objects: list, ref_idx: int):
    # return list of probabilities for related objects of reference object
    ref = objects[ref_idx]
    dist_to_ref = []
    # firstly we calculate 1/distances to reference object
    for i in range(len(objects)):
      obj = objects[i]
      if self.meet_ref_criteria(ref,obj):
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
        ref_probs = list(np.zeros(len(objects)))
        
    return list(ref_probs)
  
  def eval_pointing_param(self):
    """
    Calculates probabilities of target (GDRNet++) objects with reference
    from pointing input
    """
    dist_probs = self.evaluate_distances(self.deitic.distances_from_line)
    # print(f"[DEBUG] dist_probs: {dist_probs}")
    rows = []
    for i in range(len(self.gdrn_objects)):
      row = self.evaluate_reference(self.gdrn_objects,i)
      rows.append(list(
        dist_probs[i] * self.gdrn_objects[i]["confidence"] * np.array(row) ))
    
    probs_matrix = np.array(rows)
    # print(f"[DEBUG] prob_matrix:\n {prob_matrix}")
    ret = np.sum(probs_matrix,axis=0)
    return list(ret / np.sum(ret))
  
  def eval_pointing(self):
    """
    Calculates probabilty of target (GDRNet++) objects by their
    distance to pointing input
    """    
    def get_confidence_scores(objects: list):
      # return list of confidence score of given gdrn objects list
      scores = []
      for obj in objects:
        scores.append(obj["confidence"])
      return scores
    
    probs = self.evaluate_distances(self.deitic.distances_from_line)
    probs = np.array(probs) * np.array(get_confidence_scores(self.gdrn_objects))
    
    return list(probs)
  
  def eval_language_param(self):
    """
    Calculates probabilities of target (GDRNet++) objects with reference
    from language input
    """
    def find_indexes(objects: list, name: str):
      # returns list of object's indexes from language with same name
      ret = []
      for i in range(len(objects)):
        # strip gdrnet type and index number
        obj_name = re.match(r'^([^_]*)', objects[i]["name"][4:]).group(1)
        if obj_name == name:
          ret.append(i)
      return ret
    
    target = self.lang_objects[0] # first target object from language
    # print(target)
    indexes = find_indexes(self.lang_objects, target)
    prob = 1 / len(indexes)
    
    rows = []
    for i in indexes:
      row = self.evaluate_reference(self.gdrn_objects,i)
      rows.append(list(
        prob * self.gdrn_objects[i]["confidence"] * np.array(row) ))
    
    probs_matrix = np.array(rows)
    ret = np.sum(probs_matrix,axis=0)
    return list(ret / np.sum(ret))
  
  def eval_language(self):
    """
    Calculates probabilty of target (GDRNet++) objects by their
    """
    def find_objects(objects: list, name: str):
      # returns list of objects from language with same name
      ret = []
      for obj in objects:
        # strip gdrnet type number and index number
        obj_name = re.match(r'^([^_]*)', obj["name"][4:]).group(1)
        if obj_name == name:
          ret.append(obj)
      return ret
    
    def evaluate_objects(objects: list, sigma=self.sigma):
      # returns prob list based on object distance to robot
      unnormalized = []
      for obj in objects:
        dist = np.linalg.norm(obj["position"])
        prob = np.exp(-(dist**2) / (2 * self.sigma**2))
        unnormalized.append(prob)
      ret = unnormalized / np.sum(unnormalized)
      return list(ret)
    
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

    target_idx = np.argmax(np.array(data_dict["object_probs"]))
    self.get_logger().info("Results:")
    print(f"\tGDRN objects:\n\t {data_dict['objects']}")
    print(f"\tRespective probabilities:\n\t {data_dict['object_probs']}")
    print(f"\tTarget object with position:\n\t {data_dict['objects'][target_idx]}")
    print(f"\t {self.gdrn_objects[target_idx]['position']}\n\n")

    # Create HRICommand
    msg = HRICommand()
    msg.header = Header()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = "reasoner"

    data_str = json.dumps(data_dict)
    msg.data = [data_str]
    
    self.socket.send_string(data_str)
    self.pub.publish(msg)


def main():
  # Inputs from real modules
  rclpy.init()
  reasoner = Reasoner()
  try:
    rclpy.spin(reasoner)
  except KeyboardInterrupt:
    print("Ending by KeyboardInterrupt")
  finally:
    reasoner.destroy_node()
    rclpy.shutdown()
  
def tester():
  # function for testing
  options = ['shape', 'color', 'left', 'right']
  user_input = input(f"Select from options {options} what to test: ")
  user_input = user_input.strip().split()
  
  rclpy.init()
  reasoner = Reasoner()
  
  if user_input[0] in options:
    print(f"Option: '{user_input}' selected")
  else:
    print(f"Unknown input: '{user_input}', default selected 'color'")
    user_input = "color"
  try:
    tester = ReasonerTester(reasoner,user_input[1:],user_input[0],2.5)
    rclpy.spin(reasoner)
  except KeyboardInterrupt:
    print("Ending by KeyboardInterrupt")
  finally:
    reasoner.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
  tester()
    
