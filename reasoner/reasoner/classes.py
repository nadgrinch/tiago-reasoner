
import rclpy, zmq, ast, re, json
import numpy as np

from gesture_msgs.msg import DeicticSolution, HRICommand
from scene_msgs.msg import GDRNSolution, GDRNObject
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from rclpy.node import Node
from threading import Lock

# TODO change these topics according to inputs processing modules
DEICTIC_TOPIC = '/reasoner/input/gesture'
LANGUAGE_TOPIC = '/reasoner/input/nlp'
GDRNET_TOPIC = '/reasoner/input/gdrnet'

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
  """
  def __init__(self, node: Node, test: str):
    self.node = node
    self.test = test

    # Create test publishers
    self.pub_d = self.node.create_publisher(DeicticSolution, DEICTIC_TOPIC, 10)
    self.pub_g = self.node.create_publisher(GDRNSolution, GDRNET_TOPIC, 10)
    self.pub_n = self.node.create_publisher(HRICommand, LANGUAGE_TOPIC, 10)

    # Dummy (testing) data, tm ~ test message
    # self.tm_gdrn = self.get_gdrn_output()
    # self.tm_deictic = self.get_deictic_output()
    # self.tm_nlp = self.get_nlp_output()
    
    self.node.get_logger().info("ReasonerTester initialized")
    # self.node.get_logger().info(
    #   f"{self.tm_gdrn.objects[0].name}, {self.tm_deictic.object_name}, \n{self.tm_nlp.data}")
    
    self.timer = self.node.create_timer(5.0, self.callback)


  def callback(self):
    self.tm_gdrn = self.get_gdrn_output()
    self.tm_deictic = self.get_deictic_output()
    self.tm_nlp = self.get_nlp_output()

    self.node.get_logger().info("Publishing testing data to topics")
    self.pub_d.publish(self.tm_deictic)
    self.pub_g.publish(self.tm_gdrn)
    self.pub_n.publish(self.tm_nlp)

  def get_deictic_output(self, idx=0):
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

    deictic = DeicticSolution()
    deictic.object_id = idx
    deictic.object_name = self.tm_gdrn.objects[idx].name
    deictic.object_names = [self.tm_gdrn.objects[i].name for i in range(len(self.tm_gdrn.objects))]
    deictic.distances_from_line = calculate_distances(self.tm_gdrn.objects[idx], self.tm_gdrn)
    
    deictic.target_object_position = Point()
    deictic.target_object_position.x = float(self.tm_gdrn.objects[idx].position[0])
    deictic.target_object_position.y = float(self.tm_gdrn.objects[idx].position[1])
    deictic.target_object_position.z = float(self.tm_gdrn.objects[idx].position[2])
    
    deictic.line_point_1 = Point()
    deictic.line_point_1.x = float(deictic.target_object_position.x + 0.1)
    deictic.line_point_1.y = float(deictic.target_object_position.y)
    deictic.line_point_1.z = float(deictic.target_object_position.z + 0.2)

    deictic.line_point_2 = Point()
    deictic.line_point_2.x = float(deictic.target_object_position.x + 0.2)
    deictic.line_point_2.y = float(deictic.target_object_position.y)
    deictic.line_point_2.z = float(deictic.target_object_position.z + 0.4)

    deictic.hand_velocity = 0.0
    return deictic

  def get_gdrn_output(self):
    # currently gdrn from zmq
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
              f"[Tester] object in msg_data is not dict, but {type(msg_data)}"
              )
        gdrn.objects = objects
      else:
        self.node.get_logger().error("[Tester] Received msg_str is not list")
    except:      
      self.node.get_logger().error(
        "[Tester] Error in receiving msg from gdrn ROS1"
        )
    return gdrn

  def get_nlp_output(self):
    nlp = HRICommand()
    a_param = self.test if self.test in ["left", "right","color", "shape"] else "color"
    test_dict = {
      "action": ["pick"],
      "objects": ["null"],
      "action_param": [a_param],
      "object_param": []
    }
    nlp.header = Header()
    nlp.header.stamp = self.node.get_clock().now().to_msg()
    nlp.header.frame_id = "testingr"
    nlp.data = [json.dumps(test_dict)]

    return nlp


