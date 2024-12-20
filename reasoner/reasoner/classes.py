"""
This python file contains 3 Ros2 Subscriber classes and 1 class for testing.

PointingInput:
  DeicticSolution.msg topic subscriber
  pointting_object_selection package from Teleoperation Gesture Toolbox 
  https://github.com/imitrob/teleop_gesture_toolbox.git

LanguageInput:
  HRICommand.msg topic subscriber
  natural_gesture_processing package from iChores project
  https://github.com/ichores-research/natural_language_processing/
  HRICommand is just a way to publish dictionary of fixed language structure
  from NLP as string

GDRNInput:
  GDRNSolution.msg topic subscriber
  GDRNet++ service implementation is in repository ichores_pipeline
  https://github.com/ichores-research/ichores_pipeline/
  That uses Docker with ROS1 noetic and this is ROS2 humble, so there is need
  for some gdrnet publisher to ROS2 via ZeroMQ or make bridge from ROS1 to ROS2
  So it technically subscribes the bridged topic
"""


import rclpy, zmq, ast, re, json
import numpy as np
import random as rnd

from gesture_msgs.msg import DeicticSolution, HRICommand
from scene_msgs.msg import GDRNSolution, GDRNObject
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from rclpy.node import Node
from threading import Lock

# TODO change these topics according to inputs processing modules
DEICTIC_TOPIC = '/reasoner/input/gesture'
LANGUAGE_TOPIC = '/reasoner/input/nlp'
GDRNET_TOPIC = '/gdrn_solution'

class PointingInput():
  """
  Class for getting DeicticSolution input
  """
  def __init__(self, node: Node):
    # Storing node reference and class variables
    self.node = node
    self._lock = Lock()
    
    # ?? do I use it TODO
    self.object_id = None
    self.object_name = None
    self.names = None
    self.distances_from_line = None
    self.line_point_1 = None
    self.line_point_2 = None
    self.target_position = None
    
    # Creating subcription using parent node
    self.sub = self.node.create_subscription(
      DeicticSolution,
      DEICTIC_TOPIC,
      self.pointing_callback,
      10
    )
  
  def pointing_callback(self, msg):
    self._lock.acquire()
    self.msg = msg
    self._lock.release()
    self.node.get_logger().info("Received DeiticSolution")
  
  def get_solution(self):
    self._lock.acquire()
    solution = self.msg
    name = solution.object_name
    index = solution.object_id
    pos_x = solution.target_object_position.x
    pos_y = solution.target_object_position.y
    pos_z = solution.target_object_position.z
    self._lock.release()
    
    self.node.get_logger().info(f"Pointing at: {name}, idx: {index}")
    self.node.get_logger().info("At position: [%.4f, %.4f, %.4f]" %
                                (pos_x, pos_y, pos_z) )
    return solution
    

class LanguageInput():
  """
  Class for getting NLP input
  """
  def __init__(self, node: Node):
    # Storing node reference and class variables
    self.node = node
    self.data = None
    
    # Creating subcription using parent node
    self.sub = self.node.create_subscription(
      HRICommand,
      LANGUAGE_TOPIC,
      self.language_callback,
      10
    )
  
  def language_callback(self, msg):
    self.data = json.loads(msg.data[0])
    self.node.get_logger().info('Received HRICommand from NLP')
    
        
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
    self._lock.acquire()
    self.objects = []
    for msg_obj in msg.objects:
      obj = dict()
      obj["name"] = msg_obj.name
      obj["confidence"] = msg_obj.confidence
      obj["position"] = msg_obj.position
      obj["orientation"] = msg_obj.orientation
      obj["color"] = self.get_object_color(obj["name"])
      self.objects.append(obj)
    self._lock.release()

    self.node.get_logger().info("Received GDRNSolution")
  
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


class ReasonerTester():
  """
  Class for testing Reasoner (merger_node)
  Input:
    node: Node, required
    test: str, which action param to test
    timer_secs: float, period of timer callbacks i.e. update speed
  """
  def __init__(self, node: Node, exclude: list, test="color", timer_secs=5.0):
    self.node = node
    self.exclude = exclude
    self.test = test
    self.timer_secs = timer_secs

    self.timer = None
    self.pub_d = None
    self.pub_g = None
    self.pub_n = None
    self.sub_d = None
    self.sub_g = None
    self.sub_n = None

    off_d = True
    off_g = True
    off_n = True

    for off in self.exclude:
      if off.strip() == "deictic":
        off_d = False
      elif off.strip() == "gdrn":
        off_g = False
      elif off.strip() == "nlp":
        off_n = False
    
    self.tester_setup(self.node, off_d, off_g, off_n)
    self.node.get_logger().info("ReasonerTester initialized")

  def tester_setup(self,node: Node, deitic=True, gdrnet=True, nlp=True):
    if not deitic and not gdrnet and not nlp:
      self.pub_d = None
      self.pub_g = None
      self.pub_n = None
      return
    else:
      self.timer = node.create_timer(self.timer_secs, self.timer_callback)
    if deitic:
      self.pub_d = node.create_publisher(DeicticSolution, DEICTIC_TOPIC, 10)
    else:
      self.sub_d = node.create_subscription(
        DeicticSolution, DEICTIC_TOPIC, self.deictic_callback, 10 )
    if gdrnet:
      self.pub_g = node.create_publisher(GDRNSolution, GDRNET_TOPIC, 10)
    else:
      self.sub_g = node.create_subscription(
        GDRNSolution, GDRNET_TOPIC, self.gdrn_callback, 10 )
    if nlp:
      self.pub_n = node.create_publisher(HRICommand, LANGUAGE_TOPIC, 10)
    else:
      self.sub_n = node.create_subscription(
        HRICommand, LANGUAGE_TOPIC, self.nlp_callback, 10 )    

  def deictic_callback(self, msg):
    self.tm_deictic = msg

  def gdrn_callback(self, msg):
    self.tm_gdrn = msg

  def nlp_callback(self, msg):
    self.tm_nlp = msg

  def timer_callback(self):
    # update input data, currently only GDRNet and publish
    self.tm_gdrn = self.get_gdrn_output()
    self.tm_deictic = self.get_deictic_output()
    self.tm_nlp = self.get_nlp_output()

    # self.node.get_logger().info("Publishing testing data to topics")
    if self.pub_d:
      self.pub_d.publish(self.tm_deictic)
    if self.pub_g:
      self.pub_g.publish(self.tm_gdrn)
    if self.pub_n:
      self.pub_n.publish(self.tm_nlp)

  def get_deictic_output(self, idx=0):
    # Gets DeicticSolution input, currently dummy input
    def calculate_distances(target: GDRNObject,gdrn: GDRNSolution):
      distances = []
      t_pos = np.array(target.position)
      for obj in gdrn.objects:
        if target != obj:
          pos = np.array(obj.position)
          distances.append(float(np.linalg.norm(pos - t_pos)))
        else:
          distances.append(0.0)
      return distances

    if self.sub_d == None:
      fixed_pointing = GDRNObject()
      fixed_pointing.position = [1.13, 0.14, 0.65]

      deictic = DeicticSolution()
      deictic.distances_from_line = calculate_distances(fixed_pointing, self.tm_gdrn)
      deictic.object_id = int(np.argmax(np.array(deictic.distances_from_line)))
      deictic.object_name = self.tm_gdrn.objects[idx].name
      deictic.object_names = [self.tm_gdrn.objects[i].name for i in range(len(self.tm_gdrn.objects))]
      
      # conversion to float should not be necessary, but it caused issues
      deictic.target_object_position = Point()
      deictic.target_object_position.x = float(self.tm_gdrn.objects[idx].position[0])
      deictic.target_object_position.y = float(self.tm_gdrn.objects[idx].position[1])
      deictic.target_object_position.z = float(self.tm_gdrn.objects[idx].position[2])
      
      deictic.line_point_1 = Point()
      deictic.line_point_1.x = float(deictic.target_object_position.x + 0.2)
      deictic.line_point_1.y = float(deictic.target_object_position.y)
      deictic.line_point_1.z = float(deictic.target_object_position.z)

      deictic.line_point_2 = Point()
      deictic.line_point_2.x = float(deictic.target_object_position.x + 0.4)
      deictic.line_point_2.y = float(deictic.target_object_position.y)
      deictic.line_point_2.z = float(deictic.target_object_position.z)

      deictic.hand_velocity = 0.0
      return deictic
    return self.tm_deictic

  def get_gdrn_output(self):
    # return input from subscriber or from ZMQ
    if self.sub_g == None:
      # Creates GDRNSolution from ZeroMQ dict representation
      context = zmq.Context()
      self.socket = context.socket(zmq.SUB)
      self.socket.connect("tcp://localhost:5557")
      self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

      gdrn = GDRNSolution()
      objects = []
      try:
        msg_str = self.socket.recv_string()
        msg_data = json.loads(msg_str)
        if type(msg_data) == list:
          for obj in msg_data:
            if type(obj) == dict:
              gdrn_obj = GDRNObject()
              gdrn_obj.name = obj["name"]
              gdrn_obj.confidence = obj["confidence"]
              gdrn_obj.position = obj["position"]
              gdrn_obj.orientation = obj["orientation"]
              objects.append(gdrn_obj)
            else:
              self.node.get_logger().warn(
                f"[Tester] object in msg_data is not dict, but {type(msg_data)}")
          gdrn.objects = objects
        else:
          self.node.get_logger().error("[Tester] Received msg_str is not list")
      except:      
        self.node.get_logger().error(
          "[Tester] Error in receiving msg from gdrn ROS1")
      return gdrn
    return self.tm_gdrn

  def get_nlp_output(self):
    if self.sub_n == None:
      # Gets HRICommand from NLP, currently dummy input
      nlp = HRICommand()
      # for testing a_p ~ action_param
      a_p = self.test if self.test in ["left", "right", "shape", "color"] else ""
      test_dict = {
        "action": ["pick"],
        "objects": ["null"],
        "action_param": [a_p],
        "object_param": []
      }
      # maybe necessary to config nlp.header.stamp and frame_id
      nlp.header = Header()
      nlp.data = [json.dumps(test_dict)]

      return nlp
    return self.tm_nlp


