
import rclpy, zmq, ast, re
import numpy as np

from tiago_merger.msg import DeiticSolution, HRICommand, GDRNSolution
from rclpy.node import Node
from threading import Lock

# TODO change these topics according to inputs processing modules
DEITIC_TOPIC = '/deitic_solution'
LANGUAGE_TOPIC = '/language_output'
GDRNET_TOPIC = '/gdrnet_output'

class PointingInput():
  """
  Class for getting DeiticSolution input
  """
  def __init__(self, node: Node):
    # Storing node reference and class variables
    self.node = node
    self._lock = Lock()
    
    self.object_id = None
    self.object_name = None
    self.names = None
    self.distances_from_line = None
    self.line_point_1 = None
    self.line_point_2 = None
    self.target_position = None
    
    # Creating subcription using parent node
    self.sub = self.node.create_subscription(
      DeiticSolution,
      DEITIC_TOPIC,
      self.pointing_callback,
      10
    )
  
  def pointing_callback(self, msg):
    self.node.get_logger().info('[PointInput] Received DeiticSolution')
    self._lock.acquire()
    self.msg = msg 
    self._lock.release()    
  
  def get_solution(self):
    self._lock.release()
    solution = self.msg
    self._lock.release()
    
    return solution
    
  def evaluate_objects_by_distance(self, sigma=2.0):
    """
    Compute probability of distances_from_line using Normal distribution
    Input:
      - sigma: default value 0.3
    Output:
      - normalized_probs 
    """
    # Using Gaussian (normal) distribution
    # TODO sigma value may need to be tuned
    unnormalized_probs = []
    for distance in self.distances_from_line:
      unnormalized_probs.append(np.exp(-(distance**2)/(2 * sigma**2)))
    
    normalized_probs = unnormalized_probs / sum(unnormalized_probs)
    return normalized_probs


class LanguageInput():
  """
  Class for getting NLP input
  """
  def __init__(self, node: Node):
    # Storing node reference and class variables
    self.node = node
    
    # Creating subcription using parent node
    self.sub = self.node.create_subscription(
      HRICommand,
      LANGUAGE_TOPIC,
      self.language_callback,
      10
    )
  
  def language_callback(self, msg):
    self.node.get_logger().info('[NLPInput] Received HRICommand (string)')
    self.data = self.parse_msg(msg.data)
  
  def parse_msg(self, data):
    """
    String parser for callback function
    """

    # Remove whitespaces around colons and commas for more robust parsing
    cleaned_string = re.sub(r'\s*:\s*', ':', re.sub(r'\s*,\s*', ',', data))
    
    # Extract components
    action_match = re.search(r'action:([^,]+)', cleaned_string)
    objects_match = re.search(r'objects:\[([^\]]+)\]', cleaned_string)
    action_param_match = re.search(r'action_param:([^,]+)', cleaned_string)
    object_param_match = re.search(r'object_param:\[([^\]]+)\]', cleaned_string)
    
    # Parse components
    result = {
        "action": action_match.group(1) if action_match else None,
        "object": ast.literal_eval(f"[{objects_match.group(1)}]") if objects_match else [],
        "action_param": action_param_match.group(1) if action_param_match else None,
        "object_param": ast.literal_eval(f"[{object_param_match.group(1)}]") if object_param_match else []
    }
    
    return result
    
    
class GDRNetInput():
  """
  Class for getting GDRNet++ input
  """
  def __init__(self, node: Node):
    # Storing node reference and class variables
    self.node = node
    self._lock = Lock()

    # Creating subcription using parent node
    self.sub = self.node.create_subscription(
      GDRNSolution,
      GDRNET_TOPIC,
      self.gdrnet_callback,
      10
    )
  
  def gdrnet_callback(self, msg):
    self.node.get_logger().info("[GDRNetInput] Received GDRNSolution")
    self.objects = []
    
    self._lock.acquire()
    for msg_obj in msg.objects:
      obj = dict()
      obj["name"] = msg_obj.name
      obj["confidence"] = msg_obj.confidence
      obj["position"] = msg_obj.position
      obj["orientation"] = msg_obj.orientation
      obj["color"] = self.get_object_color(obj)
      self.objects.append(obj)
    
    self._lock.release()
  
  def get_objects(self):
    self._lock.acquire()
    gdrn_objects = self.objects
    self._lock.release()
    return gdrn_objects
  
  def get_object_color(self, name: str):
    # TODO this function is hardcoding color and other param
    # to detected objects from known properties
    # needs to be changed so it reads ontology
    color = {
      "001_chips_can" : "red",
      "002_master_chef_can" : "blue",
      "003_cracker_box"  : "red",
      "004_sugar_box"  : "yellow",
      "005_tomato_soup_can"  : ["white","red"],
      "006_mustard_bottle"  : "yellow",
      "008_pudding_box"  : "brown",
      "009_gelatin_box"  : "red",
      "010_potted_meat_can"  : "blue",
      "011_banana"  : "yellow",
      "013_apple"  : "red",
      "014_lemon"  : "yellow",
      "015_peach"  : "orange",
      "016_pear"  : "green",
      "017_orange"  : "orange",
      "018_plum"  : "purple",
      "021_bleach_cleanser"  : "white",
      "024_bowl"  : "red",
      "025_mug"  : "red",
      "029_plate"  : "red",
    }
    object_color = None
    name_key = name[:len(name)-2] # remove indexing from name
    try:
      if name_key not in color.keys():
        self.node.get_logger().warn(
          f"[GDRNetInput] object name {name} not in color dict!"
        )
      else:
        object_color = color[name_key]
    except:
      self.node.get_logger().error(
        "[GDRNetInput] Error in accesing object['name'] in get_object_color function!"
      )

    return object_color


