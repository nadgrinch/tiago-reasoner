# Reasoner
This ROS2 (humble) package should merge together 3 inputs from previous modules from ichores pipeline:
- Gesture (human detection)
  ```
  deitic_solution = {
    "object_id": int,
    "object_name": string,
    "object_names": [string, string, ...],
    "distances_from_line": [float, float, ...],
    "line_point_1": Point ~ [float, float, float],
    "line_point_2": Point,
    "target_object_position": Point
    }
  ```
- Language parser (speech recognition) *(string output)*
  ```
  action: <name>,
  objects: [ <object_id1>, <object_id2> ],
  action_param: <parameter>
  object_param: [ [ <parameter_o1_1>, <parameter_o1_2> ],
                [ <parameter_o2_1>, <parameter_o2_2> ] ]
  ```
- Scene (object detection) *gdrnet++*
  ```
  [ object_id1 = {
      "name": string,
      "confidence": float,
      "position": [float, float, float],
      "orientation": [float, float, float, float],
    },
    object_id2 = {
      "name": string,
      "confidence": float,
      "position": [float, float, float],
      "orientation": [float, float, float, float],
    },
    ... ]
  ```
  *technically gdrnet outputs List of objects 'PoseWithConfidence', but for simplification, in this project the output is parsed to Dictionary*

To one out as HRI command:
- HRI command
  ```
  Header header
  string[] data

  # Example data
  # data = {
  #   "action": ["pick], "action_probs": [1.0],
  #   "objects": ["banana", "apple"], "object_probs": [0.7, 0.3]
  # }
  ``` 
## Merging
*What does this module do?*

Main task that this module is doing is merging all 3 inputs, so referencies in language are filled with information from gesture and objects detection. In a example action for robot controller from sentence `Pick same shape as this one.` will be enhanced by switching `this` for the most probable object that the person is pointing at. Also the shape for which we are looking for in other objects in scene will be read from ontology, in base version GDRNet++.

During process of merging probabilities of object detection or object selection or probability of said words are propagated. So the final output is action with probability of intent, which we by going through all detected objects, for each we compute distances to other objects that meet the filtering criteria. From these distances probability is calculated from formula `P = (sum(distances) - distance) / sum(distances)` if more that 1 object meets criteria. This prob. is then multiplyed by GDRNet confidence score and corresponding Pointing probability. Finally we add all probabilities of each object and normalize them.

*The process of computing probability is demostrated in `prob_demo.py` script*

## Testing
Currently in the `classes.py` is class `ReasonerTester`, which creates dummy data or takes data from topics of previous modules in pipeline to test **Reasoner** functionality. This class and it usage is in early develompment, so documentation only in code by describing what each part does. If described (: *December 2024*
