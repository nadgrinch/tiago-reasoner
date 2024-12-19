#!/usr/bin/python3

import rclpy, zmq, re, json, time
import numpy as np

from gesture_msgs.msg import HRICommand, DeicticSolution
from scene_msgs.msg import GDRNObject, GDRNSolution
from std_msgs.msg import Header
from reasoner.classes import PointingInput, LanguageInput, GDRNetInput, ReasonerTester
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
    context = zmq.Context()
    self.socket = context.socket(zmq.PUB)
    self.socket.bind("tcp://*:5558")
    
    # Create timer for main processing loop
    self.timer = self.create_timer(2.0, self.main_loop)
    
    self.get_logger().info('Tiago Merger node initialized')
        
  def main_loop(self):
    # Main processing loop
    if self.language_sub.data == None:
      self.get_logger().warn("LanguageInput empty, skipping calculations!")
      return
    
    self.action = self.language_sub.data["action"]
    self.action_param = self.language_sub.data["action_param"][0]
    self.lang_objects = self.language_sub.data["objects"]
    self.lang_objects_param = self.language_sub.data["object_param"]
    
    self.deitic = self.pointing_sub.get_solution()
    self.gdrn_objects = self.gdrnet_sub.get_objects()
    
    self.get_logger().info(f"[tiago_reasoner_node] Action parameter: {self.action_param}")
    if self.action_param in self.relations:
      # if known action param and if target objects
      if self.nlp_empty():
        self.get_logger().info("Pointing ref")
        self.target_probs = self.eval_pointing_ref()
      else:
        self.get_logger().info("Language ref")
        self.target_probs = self.eval_language_ref()
        pass
      
    else:
      if self.nlp_empty():
        # pick object human is pointing at
        self.get_logger().info("Pointing")
        self.target_probs = self.eval_pointing()
      else:
        self.get_logger().info("Language")
        self.target_probs = self.eval_language()
    
    self.publish_results()

  def wait(self, duration:int):
    # simple wait duration number of rates
    self.get_logger().info(f"Waiting for {duration} seconds...")
    time.sleep(duration)
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
        stripped = re.match(r'^([^_]*)', name[4:]).group(1) # strip gdrnet index number
        # all the above makes from 011_banana_1, banana
        if target == stripped:
          ret = False
      
    return ret

  def meet_ref_criteria(self, ref: dict, obj: dict, tolerance=0.1, scale=10.0):
    # return True if given object meets filter criteria of reference
    ret = False
    ref_num = int(ref["name"][:3])
    obj_num = int(obj["name"][:3])
    # TODO better color and id ==
    if (self.action_param == "color" and obj["color"] == ref["color"]):
      ret = True
    elif (self.action_param == "shape" and obj_num == ref_num):
      ret = True
    elif self.action_param in ["left", "right"]:
      dir_vector = [
        scale*(self.deitic.line_point_1.x - self.deitic.line_point_2.x), 
        scale*(self.deitic.line_point_1.y - self.deitic.line_point_2.y)
        ]
      # print(dir_vector)
      ref_pos = ref["position"]
      obj_pos = obj["position"]
      
      ref_to_pos = [ref_pos[0] - obj_pos[0], ref_pos[1] - obj_pos[1]]
      dot_product = (
        ref_to_pos[0] * dir_vector[1] + 
        ref_to_pos[1] * -dir_vector[0]
        )

      print(f"{ref_pos},{obj_pos},{dot_product}")
      if self.action_param == "left" and dot_product < -tolerance:
        ret = True
      elif self.action_param == "right" and dot_product > tolerance:
        ret = True
    
    # print(f"return: {ret}, ref, obj: {ref_num}, {obj_num}")
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
        print(ref_obj["name"],obj["name"])
        if self.meet_ref_criteria(ref_obj,obj):
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

    # self.get_logger().info(f"distances: {self.deitic.distances_from_line}")
    dist_probs = evaluate_distances(self.deitic.distances_from_line,sigma=2.0)
    # self.get_logger().info(f"dist_probs: {dist_probs}\n\n")
    rows = []
    for i in range(len(self.gdrn_objects)):
      row = evaluate_reference(self.gdrn_objects,i)
      print(row)
      rows.append(list(
        dist_probs[i] * self.gdrn_objects[i]["confidence"] * np.array(row)
        ))
    
    probs_matrix = np.array(rows)
    # print(probs_matrix)
    ret = np.sum(probs_matrix,axis=0)
    return list(ret / np.sum(ret))
  
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
    
    return list(probs)
  
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
    # print(target)
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
    return list(ret / np.sum(ret))
  
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
    self.get_logger().info(f"""\nTarget objets:\n {data_dict['objects']}
                           \rProbs results:\n {data_dict['object_probs']}""")

    # Create HRICommand
    msg = HRICommand()
    msg.header = Header()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = "reasoner"

    msg.data = [json.dumps(data_dict)]
    
    serialized_msg = json.dumps({
        "header": {
            "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
            "frame_id": msg.header.frame_id
        },
        "data": msg.data
    })
    self.socket.send_string(serialized_msg)
    self.pub.publish(msg)
    self.wait(10)


def main():
  # better user interface for testing
  rclpy.init()
  merger_node = Reasoner()
  tester = ReasonerTester(merger_node,"left")
  rclpy.spin(merger_node)
  merger_node.destroy_node()
  rclpy.shutdown()
    

if __name__ == '__main__':
  main()
    
