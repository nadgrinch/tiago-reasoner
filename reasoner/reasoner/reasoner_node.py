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
    self.published = False
    self.sigma = sigma
    self.lock = Lock()
    
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
    if self.language_sub.data == None and not self.get_published():
      self.get_logger().warn("Nothing (new) received from inputs, waiting")
      return
    
    self.action = self.language_sub.data["action"]
    self.action_param = self.language_sub.data["action_param"][0]
    self.lang_objects = self.language_sub.data["objects"]
    self.lang_objects_param = self.language_sub.data["object_param"]
    
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

  def get_published(self):
    self.lock.acquire()
    published = self.published
    self.lock.release()
    return published

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

  def eval_pointing_param(self):
    """
    Calculates probabilities of target (GDRNet++) objects with reference
    from pointing input
    """
    # demo_prob.py implementation
    
    def evaluate_distances(distances: list):
      # returns list of probabilities from distances
      # distances to Gaussian distribution and normalized
      unnormalized_p = []
      for dist in distances:
          prob = np.exp(-(dist**2) / (2 * self.sigma**2))
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
        # print(ref_obj["name"],obj["name"])
        if self.meet_ref_criteria(ref_obj,obj):
          d = np.linalg.norm(
            np.array(obj["position"]) - np.array(ref_obj["position"]) )
          # distance d ~ smaller better, Gaussian distribution
          if d != 0.0:
            d_prob = np.exp(-(d**2) / (2 * 2.0**2))
          else:
            d_prob = d
          target_probs.append(d_prob)
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

    dist_probs = evaluate_distances(self.deitic.distances_from_line)
    # print(f"[DEBUG] dist_probs: {dist_probs}")

    # print(self.deitic.line_point_1.x - self.deitic.line_point_2.x,
    #       self.deitic._line_point_1.y - self.deitic.line_point_2.y)

    rows = []
    for i in range(len(self.gdrn_objects)):
      row = evaluate_reference(self.gdrn_objects,i)
      # print(row)
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
    
    def evaluate_distances(distances: list):
      # returns list of probabilities from distances
      # distances to Gaussian distribution and normalized
      unnormalized_p = []
      for dist in distances:
          prob = np.exp(-(dist**2) / (2 * self.sigma**2))
          unnormalized_p.append(prob)
      normalized_p = unnormalized_p / np.sum(unnormalized_p)
      return normalized_p
    
    def get_confidence_scores(objects: list):
      # return list of confidence score of given gdrn objects list
      scores = []
      for obj in objects:
        scores.append(obj["confidence"])
      return scores
    
    probs = evaluate_distances(self.deitic.distances_from_line)
    probs = np.array(probs) * np.array(get_confidence_scores(self.gdrn_objects))
    
    return list(probs)
  
  def eval_language_param(self):
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
            np.array(ref_obj["position"]) - np.array(obj["position"]) )
          # distance d ~ smaller better, Gaussian distribution
          if d != 0.0:
            d_prob = np.exp(-(d**2) / (2 * 2.0**2))
          else:
            d_prob = d
          target_probs.append(d_prob)
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
    # print(target)
    indexes = find_indexes(self.lang_objects, target)
    prob = 1 / len(indexes)
    
    rows = []
    for i in indexes:
      row = evaluate_reference(self.gdrn_objects,i)
      rows.append(list(
        prob * self.gdrn_objects[i]["confidence"] * np.array(row) ))
    
    probs_matrix = np.array(rows)
    ret = np.sum(probs_matrix,axis=0)
    return list(ret / np.sum(ret))
  
  def eval_language(self):
    """
    Calculates probabilty of target (GDRNet++) objects by their
    """
    def find_objects(objects: list, param: str):
      # returns list of object's indexes from language
      ret = []
      for obj in objects:
        # strip gdrnet type number and index number
        obj_name = re.match(r'^([^_]*)', obj["name"][4:]).group(1)
        if obj_name == param:
          ret.append(obj)
      return ret
    
    def evaluate_objects(objects: list):
      # returns prob list based on object distance to robot
      unnormalized = []
      for obj in objects:
        dist = np.linalg.norm(obj["position"])
        prob = np.exp(-(dist**2) / (2 * self.sigma**2))
        unnormalized.append(prob)
      ret = unnormalized / np.sum(unnormalized)
      return list(ret)
    
    language_objects = find_objects(self.gdrn_objects, self.lang_objects[0])
    probs = evaluate_objects(language_objects)
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
  try:
    reasoner = Reasoner(sigma=float(user_input[1]))
  except:
    reasoner = Reasoner()
  
  if user_input[0] in options:
    print(f"Option: '{user_input}' selected")
  else:
    print(f"Unknown input: '{user_input}', default selected 'color'")
    user_input = "color"
  try:
    tester = ReasonerTester(reasoner,user_input[0],5.0)
    rclpy.spin(reasoner)
  except KeyboardInterrupt:
    print("Ending by KeyboardInterrupt")
  finally:
    reasoner.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
  tester()
    
