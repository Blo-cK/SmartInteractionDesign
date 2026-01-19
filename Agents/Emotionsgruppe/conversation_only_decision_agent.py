from data_models.data_models import AgentState, NextActionDecision, NextActionDecisionType
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent
from large_language_models.llm_factory import llm_factory
from langchain_core.prompts import ChatPromptTemplate
import requests
import json
from datetime import datetime
import os

class SensorData():

    def __init__(self):
        self.newest_gaze_data = None
        self.last_gaze_timestamp = None
        self.newest_head_gesture_data = None
        self.last_head_gesture_timestamp = None
        self.newest_face_emotion_data = None
        self.newest_text_emotion_data = None
        self.last_face_emotion_timestamp = None
        self.last_text_emotion_timestamp = None
        self.face_emotion_buffer = []  # List of recent face emotion readings
        self.text_emotion_buffer = []  # List of recent text emotion readings
        self.used_text_emotion_timestamps = set()  # Track which text emotions have been used
        self.used_face_emotion_timestamps = set()  # Track which face emotions have been used
        self.last_face_emotion_timestamp = None
        self.last_text_emotion_timestamp = None
        
    def isGazeDetectionAvailable(self):
        response = requests.get("http://152.53.32.66:5000/api/services/input/monitor/camera1.fullframe")
        print("is gazedetection service available?", response.json())
        data = response.json()
        online_status = data.get("camera1.fullframe", {}).get("online", False)
        print("Gaze detection service online:", online_status)
        return online_status

    def get_gaze_detection_data(self):    
        response = requests.get("http://152.53.32.66:5000/api/history/gaze_detector")
        data = response.json()
        return data
    
    def get_last_gaze_timestamp(self):
        return self.last_gaze_timestamp
    
    def isNewDataAvailableGaze(self):
        if self.isGazeDetectionAvailable() == False:
            return False
        data = self.get_gaze_detection_data()
        if data and len(data) > 0:
            print(data[-1]["time_stamp"], " vs ", self.last_gaze_timestamp)
            # Check if this is first call or if timestamp has changed
            if self.last_gaze_timestamp is None or data[-1]["time_stamp"] != self.last_gaze_timestamp:
                self.newest_gaze_data = data[-1]
                self.last_gaze_timestamp = self.newest_gaze_data["time_stamp"]
                return True
            return False
        return False
    
    def headGestureAvailable(self):
        response = requests.get("http://152.53.32.66:5000/api/services/input/monitor/camera1.fullframe")
        print("Is the head gesture service available?", response.json())
        data = response.json()
        online_status = data.get("camera1.fullframe", {}).get("online", False)
        print("Head gesture service online:", online_status)
        return online_status

    def get_head_gesture_data(self):    
        response = requests.get("http://152.53.32.66:5000/api/history/headgesture_recognition")
        data = response.json()
        return data
    
    def get_last_head_gesture_timestamp(self):
        return self.last_head_gesture_timestamp
    
    def isNewDataAvailableHeadGesture(self):
        if self.headGestureAvailable() == False:
            return False
        data = self.get_head_gesture_data()
        print(f"DEBUG Head Gesture: API returned {len(data) if data else 0} items: {data}")
        if data and len(data) > 0:
            print(data[-1]["time_stamp"], " vs ", self.last_head_gesture_timestamp)
            # Check if this is first call or if timestamp has changed
            if self.last_head_gesture_timestamp is None or data[-1]["time_stamp"] != self.last_head_gesture_timestamp:
                self.newest_head_gesture_data = data[-1]
                self.last_head_gesture_timestamp = self.newest_head_gesture_data["time_stamp"]
                return True
            return False
        return False
    
    BUFFER_SIZE = 5  # Keep last 5 emotion readings
    TIMESTAMP_TOLERANCE_MS = 2000  # Consider emotions from within 2 seconds as corresponding

    def get_face_emotion_data(self):    
        response = requests.get("http://152.53.32.66:5000/api/history/face_emotion")
        data = response.json()
        return data
    
    def get_text_emotion_data(self):    
        response = requests.get("http://152.53.32.66:5000/api/history/text_emotion")
        data = response.json()
        return data

    def get_last_text_emotion_timestamp(self):
        return self.last_text_emotion_timestamp
    
    def get_last_face_emotion_timestamp(self):
        return self.last_face_emotion_timestamp
    
    def _update_buffer(self, buffer, new_data):
        """Update buffer, keeping only the most recent BUFFER_SIZE items."""
        buffer.append(new_data)
        while len(buffer) > self.BUFFER_SIZE:
            buffer.pop(0)
    
    def _is_timestamp_close(self, timestamp1, timestamp2):
        """Check if two timestamps are within tolerance (milliseconds)."""
        if timestamp1 is None or timestamp2 is None:
            return False
        try:
            diff = abs(float(timestamp1) - float(timestamp2))
            return diff <= self.TIMESTAMP_TOLERANCE_MS
        except (TypeError, ValueError):
            return False
    
    def get_unused_text_emotions(self):
        """Return text emotions that haven't been used yet, sorted by newest first."""
        unused = [d for d in self.text_emotion_buffer 
                  if d.get("time_stamp") not in self.used_text_emotion_timestamps]
        return sorted(unused, key=lambda x: x.get("time_stamp", 0), reverse=True)
    
    def get_corresponding_face_emotion(self, text_emotion_timestamp):
        """Find face emotion data with closest matching timestamp."""
        if not self.face_emotion_buffer:
            return None
        
        closest = min(
            self.face_emotion_buffer,
            key=lambda x: abs(float(x.get("time_stamp", 0)) - float(text_emotion_timestamp or 0))
        )
        
        if self._is_timestamp_close(closest.get("time_stamp"), text_emotion_timestamp):
            return closest
        return None
    
    def mark_text_emotion_as_used(self, timestamp):
        """Mark a text emotion as used."""
        if timestamp is not None:
            self.used_text_emotion_timestamps.add(timestamp)
    
    def mark_face_emotion_as_used(self, timestamp):
        """Mark a face emotion as used."""
        if timestamp is not None:
            self.used_face_emotion_timestamps.add(timestamp)
    
    def is_new_face_emotion_data_available(self):
        data = self.get_face_emotion_data()
        if data and len(data) > 0:
            # Check if this is first call or if timestamp has changed
            if self.last_face_emotion_timestamp is None or data[-1]["time_stamp"] != self.last_face_emotion_timestamp:
                self.newest_face_emotion_data = data[-1]
                self.last_face_emotion_timestamp = self.newest_face_emotion_data["time_stamp"]
                self._update_buffer(self.face_emotion_buffer, data[-1])
                return True
            return False
        return False    
    
    def is_new_text_emotion_data_available(self):
        data = self.get_text_emotion_data()
        if data and len(data) > 0:
            # Check if this is first call or if timestamp has changed
            if self.last_text_emotion_timestamp is None or data[-1]["time_stamp"] != self.last_text_emotion_timestamp:
                self.newest_text_emotion_data = data[-1]
                self.last_text_emotion_timestamp = self.newest_text_emotion_data["time_stamp"]
                self._update_buffer(self.text_emotion_buffer, data[-1])
                return True
            return False
        return False    

class ConversationOnlyDecisionAgent(BaseDecisionAgent):

    def __init__(self):
        super().__init__()
        self.llm = llm_factory.get_llm()
        self.sensordata = SensorData()
        self.log_directory = "sensor_logs"
        os.makedirs(self.log_directory, exist_ok=True)

    def next_action(self, agent_state: AgentState):

        next_action_decision = NextActionDecision(
            type=NextActionDecisionType.GENERATE_ANSWER,
            action=None,
            payload=None
        )
        
        # Check each sensor data type individually
        gaze_available = self.sensordata.isNewDataAvailableGaze()
        head_gesture_available = self.sensordata.isNewDataAvailableHeadGesture()
        face_emotion_available = self.sensordata.is_new_face_emotion_data_available()
        text_emotion_available = self.sensordata.is_new_text_emotion_data_available()
        
        sensor_data_is_available = gaze_available or head_gesture_available or face_emotion_available or text_emotion_available
        
        print("Individual sensor checks - Gaze:", gaze_available, "| Head Gesture:", head_gesture_available, "| Face Emotion:", face_emotion_available, "| Text Emotion:", text_emotion_available)
        print("Sensor data available:", sensor_data_is_available, " | Last timestamp:", self.sensordata.get_last_gaze_timestamp(), " | Newest gaze data:", self.sensordata.newest_gaze_data, " | Last head gesture timestamp:", self.sensordata.get_last_head_gesture_timestamp(), 
              " | Newest head gesture data:", self.sensordata.newest_head_gesture_data, " | Last face emotion timestamp:", self.sensordata.get_last_face_emotion_timestamp(), " | Newest face emotion data:", self.sensordata.newest_face_emotion_data, " | Last text emotion timestamp:", 
              self.sensordata.get_last_text_emotion_timestamp(), " | Newest text emotion data:", self.sensordata.newest_text_emotion_data)

        
        #If new sensor data is available, the prompt will add the sensor information
        person_information = "you do not have any information about the person you are talking to."
        
        if sensor_data_is_available:            
            person_information = str(self.sensordata.newest_gaze_data) + " " + str(self.sensordata.newest_head_gesture_data) + " " + str(self.sensordata.newest_face_emotion_data) + " " + str(self.sensordata.newest_text_emotion_data)
            print("\n\nPerson information from sensors:", person_information + "\n\n")
            
            # Save person information to file
            self._save_person_information_to_file({
                "timestamp": datetime.now().isoformat(),
                "gaze_data": self.sensordata.newest_gaze_data,
                "head_gesture_data": self.sensordata.newest_head_gesture_data,
                "face_emotion_data": self.sensordata.newest_face_emotion_data,
                "text_emotion_data": self.sensordata.newest_text_emotion_data
            })
              
            environment_information = "You are in a room and looking at an embodied agent."

            # Aggregate sensor data with LLM. Later this service can be hosted externally
            person_and_environment_description = self.aggregate_sensor_information_with_llm(agent_state, person_information, environment_information)
                       
            #print("LLM-aggregated sensor information:", person_and_environment_description)

            next_action_decision = NextActionDecision(
                type=NextActionDecisionType.PROMPT_ADAPTION,
                action="sensor",
                payload={
                    "sensor_information": person_and_environment_description
                }            
            )
                
        return next_action_decision
    

    def aggregate_sensor_information_with_llm(self, agent_state, person_information, environment_information):
        
        sensor_data_aggregation_prompt = """
The following conversation between you and me => do not use the history for any gesture or emotion detection:
{chat_history}
Here are the informations from the sensors and sources:
{sensor_data}
Write a short text describing the person (gaze and head position) and their surroundings.
You are detecting a person with the following head rotation and head position. Yaw (-90° to +90°) 
is the horizontal head rotation (left/right), pitch (-90° to +90°) is the vertical tilt (up/down), 
roll (-180° to +180°) is the lateral head tilt (shoulder), and head_position (x: 0.0-1.0, y: 0.0-1.0) 
are the normalised 2D coordinates of the tip of the nose in the frame. 
For head_position, x=0.0 is left and x=1.0 is right; y=0.0 is top and y=1.0 is bottom (standard image coordinates). 
You are also detecting some Head Gestures like NOD, SHAKE, TURN, UP DOWN, TURN, WAGGLE. 
Use this information to create a brief description of the person and the situation.
You are also given information about the detected emotions of the person from facial expression analysis.

If the person is for example looking heavily away from you, 
you have to mention that the person is not fully focused on the conversation.
Change your speech and how you interact based on the current emotions the person interacting with you exhibits.
"""
            
        prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant standing in the center of the screen (x: 0.5, y: 0.5), who receives information about the person you are talking to and the current situation via various sensors and sources. Your task is to create a brief description of the person and the situation so that this information can be used in the conversation to make it more context-sensitive."),
                ("human", sensor_data_aggregation_prompt)
            ])

        self.chain = prompt | self.llm            

        response = self.chain.invoke(
                {
                    "chat_history": agent_state.chat_history,
                    "sensor_data": "\n".join([person_information,environment_information])
                }
            )

        person_and_environment_description = response.content

        return person_and_environment_description
    
    def _save_person_information_to_file(self, data):
        """Save person information to a JSON log file (overwrites existing file)."""
        try:
            log_file = os.path.join(self.log_directory, "person_information_log.json")
            
            # Overwrite file with new data
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving person information to file: {e}")
