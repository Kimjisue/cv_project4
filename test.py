import cv2
import mediapipe as mp
import math
from enum import Enum
import time

import os
import subprocess
import pyautogui

import platform

myOsName = platform.system()

# 화면 크기 가져오기
screen_width, screen_height = pyautogui.size()

# 마우스 제어 함수
def move_mouse(x, y):
    pyautogui.moveTo(x, y)

def click_mouse():
    pyautogui.click()

def volume_up():
    print("volume up")
    if(myOsName == "Darwin") :
      subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 10)"])

def volume_down():
    print("volume down")
    if(myOsName == "Darwin") :
      subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 10)"])

def volume_control():
    if(myOsName == "Darwin") :
      percent = max(0, min(100, volumePercent))
      volume_level = int(percent)
      subprocess.run([ "osascript", "-e", f"set volume output volume {volume_level}"])

def switch_app_back():
    if(myOsName == "Darwin") :
      subprocess.run(["osascript", "-e", "tell application \"System Events\" to key code 123 using control down"])

def switch_app_front():
    if(myOsName == "Darwin") :
      subprocess.run(["osascript", "-e", "tell application \"System Events\" to key code 124 using control down"])

def zoom_in():
    if(myOsName == "Darwin") :
      subprocess.run(["osascript", "-e", "tell application \"System Events\" to key code 24 using {command down}"])
      subprocess.run(["osascript", "-e", "tell application \"System Events\" to key code 24 using {command down}"])

def zoom_out():
    if(myOsName == "Darwin") :
      subprocess.run(["osascript", "-e", "tell application \"System Events\" to key code 27 using {command down}"])
      subprocess.run(["osascript", "-e", "tell application \"System Events\" to key code 27 using {command down}"])

def mute_toggle():
    print("Toggling Mute")
    if(myOsName == "Darwin") :
      subprocess.run(["osascript", "-e", "set volume with output muted"])

def scroll_up():
  pyautogui.scroll(10) 

def scroll_down():
  pyautogui.scroll(-10) 

# 오른손 제스처 모드 정의 (Right hand gesture modes)
class RightMode(Enum):
  Nothing = 0,
  Pointer = 1,
  UpScroll = 2,
  DownScroll = 3,
  Zoom = 4,

# 왼손 제스처 모드 정의 (Left hand gesture modes)
class LeftMode(Enum):
  Nothing = 0,
  Volume = 1,
  BnFReady = 2,
  Mute = 3,


# 동작 정의 (Actions for both hands)
class Action(Enum):
  Nothing = 0,
  Cursor = 1,
  Click = 2,
  UpScroll = 3,
  DownScroll = 4,
  ZoomReady = 5,
  ZoomIn = 6,
  ZoomOut = 7,
  VolumeUp = 8,
  VolumeDown = 9,
  VolumeControl = 10,
  BnFReady = 11,
  Back = 12,
  Front = 13,
  Mute = 14

action_to_function = {
    Action.VolumeUp: volume_up,
    Action.VolumeDown: volume_down,
    Action.VolumeControl: volume_control,
    Action.Mute: mute_toggle,
    Action.Cursor: lambda: move_mouse(screen_x, screen_y),
    Action.Click: click_mouse,
    Action.Back: switch_app_back,
    Action.Front: switch_app_front,
    Action.ZoomIn: zoom_in,
    Action.ZoomOut: zoom_out,
    Action.UpScroll: scroll_up,
    Action.DownScroll: scroll_down
}


# 선택된 랜드마크들을 포함하는 사각형의 최소/최대 좌표 계산
# Calculate the bounding box (rect) of selected landmarks
def rectRangeOfPoints(x_list, y_list, selected_indices):
  x_min = min([x_list[i] for i in selected_indices])
  x_max = max([x_list[i] for i in selected_indices])
  y_min = min([y_list[i] for i in selected_indices])
  y_max = max([y_list[i] for i in selected_indices])

  return [(x_min, y_min), (x_max, y_max)]

# 사각형을 기준으로 원의 중심 좌표와 반지름 계산
# Calculate the circle's center and radius from the rectangle
def circleRangeOfPoints(rect_range):
  # 중심 좌표, center
  cx = (rect_range[0][0] + rect_range[1][0]) // 2
  cy = (rect_range[0][1] + rect_range[1][1]) // 2

  # width = rect_range[1][0] - rect_range[0][0]
  # height = rect_range[1][1] - rect_range[0][1]

  # radius = max(width, height) // 2

  # 원의 반지름 계산, radius
  radius = int(math.sqrt((rect_range[0][0] - cx) ** 2 + (rect_range[0][1] - cy) ** 2))
  return [(cx, cy), radius]

# 선택된 랜드마크들이 원 안에 있는지 확인
# Check if selected landmarks are inside the circle
def isInCircleRange(circle_point, selected_points):
  cx, cy = circle_point[0]  # 중심 좌표, center point
  radius = circle_point[1]  # 반지름, radius

  isInCircle = True
  for points in selected_points:
    x, y = points
    distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if not distance <= radius:
      isInCircle = False
      break

  return isInCircle

# 오른손 제스처 모드 결정 (Determine right-hand gesture mode)
def controlRightMode(isFingersInCircle):
  if isFingersInCircle[0] and not isFingersInCircle[1] and isFingersInCircle[2] and isFingersInCircle[3] and isFingersInCircle[4]:
    return RightMode.Pointer
  if not isFingersInCircle[0] and isFingersInCircle[1] and isFingersInCircle[2] and isFingersInCircle[3] and not isFingersInCircle[4]:
    return RightMode.UpScroll
  if isFingersInCircle[0] and not isFingersInCircle[1] and not isFingersInCircle[2] and not isFingersInCircle[3] and isFingersInCircle[4]:
    return RightMode.DownScroll
  if isFingersInCircle[0] and isFingersInCircle[1] and isFingersInCircle[2] and isFingersInCircle[3] and isFingersInCircle[4]:
    return RightMode.Zoom
  return RightMode.Nothing

# 왼손 제스처 모드 결정 (Determine left-hand gesture mode)
def controlLeftMode(isFingersInCircle):
  if not isFingersInCircle[1] and not isFingersInCircle[2] and not isFingersInCircle[3]: 
    return LeftMode.BnFReady
  if not isFingersInCircle[0] and not isFingersInCircle[1] and isFingersInCircle[2] and isFingersInCircle[3] and isFingersInCircle[4]:
    return LeftMode.Volume
  if isFingersInCircle[0] and not isFingersInCircle[1] and isFingersInCircle[2] and isFingersInCircle[3] and isFingersInCircle[4]:
    return LeftMode.Mute
  return LeftMode.Nothing

# 오른손 모드 상태 안정화 (Stabilize right-hand mode state)
def rightStableModeState(new_state):
    global current_right_mode, previous_right_mode, right_change_time

    # 새로운 상태가 이전 상태와 다르면 시간 초기화
    if new_state != previous_right_mode:
        previous_right_mode = new_state
        right_change_time = time.time()  # 현재 시각으로 초기화

    # elapsed_time = time.time() - state_change_time
    # remaining_time = max(0, STATE_HOLD_DURATION - elapsed_time)
    
    # 새로운 상태가 1초 동안 유지되었으면 상태 업데이트
    if time.time() - right_change_time >= STATE_HOLD_DURATION:
      current_right_mode = new_state

# 왼손 모드 상태 안정화 (Stabilize left-hand mode state)
def leftStableModeState(new_state):
  global current_left_mode, previous_left_mode, left_change_time

  # 새로운 상태가 이전 상태와 다르면 시간 초기화
  if new_state != previous_left_mode:
      previous_left_mode = new_state
      left_change_time = time.time()  # 현재 시각으로 초기화

  # elapsed_time = time.time() - state_change_time
  # remaining_time = max(0, STATE_HOLD_DURATION - elapsed_time)
  
  # 새로운 상태가 1초 동안 유지되었으면 상태 업데이트
  if time.time() - left_change_time >= STATE_HOLD_DURATION:
    current_left_mode = new_state

def controlPointer(isFingersInCircle):
  global last_pointer_action_time
  if time.time() - last_pointer_action_time > 1:
    if isFingersInCircle[1]:
      return Action.Nothing
    elif not isFingersInCircle[2]: 
      last_pointer_action_time = time.time()
      return Action.Click 
    else:
      return Action.Cursor 
  else:
    return Action.Cursor 


def controlZoom(isFingersInCircle, action):
  global isZoom, zoom_change_time
  if time.time()-zoom_change_time < 1:
    # if isZoom:
    #   return Action.ZoomIn
    # else :
    #   return Action.ZoomOut
    return action
  elif not isFingersInCircle[0] and not isFingersInCircle[1] and isFingersInCircle[2] and isFingersInCircle[3] and isFingersInCircle[4]:
    if not isZoom:
      isZoom = True
      zoom_change_time = time.time()
      return Action.ZoomIn
    else:
      isZoom = False
      zoom_change_time = time.time()
      return Action.ZoomOut
  return Action.ZoomReady
  
def controlScroll():
    global last_scroll_action_time

    # 0.5초 간격으로 실행 제한
    if time.time() - last_scroll_action_time > 0.5:
        last_scroll_action_time = time.time()
        return Action.UpScroll
    else:
        return Action.Nothing

def controlDownScroll():
    global last_down_scroll_action_time

    # 0.5초 간격으로 실행 제한
    if time.time() - last_down_scroll_action_time > 0.5:
        last_down_scroll_action_time = time.time()
        return Action.DownScroll
    else:
        return Action.Nothing

# def controlVolume(y_list, screen_height, isFingersInCircle):
#   global previous_volume_position, current_volume_position, volume_change_time

#   if isFingersInCircle[1] and isFingersInCircle[2]:
#     return Action.Nothing

#   thumb_tip_y = y_list[mp.solutions.hands.HandLandmark.THUMB_TIP.value]

#   top_bound = screen_height // 2
#   # top_bound = screen_height // 3
#   # bottom_bound = 2 * (screen_height // 3)

#   if thumb_tip_y < top_bound:
#     current_volume_position = "Top"
#   elif thumb_tip_y > top_bound:
#     current_volume_position = "Bottom"

#   if current_volume_position != previous_volume_position:
#     if time.time() - volume_change_time < 1:
#       if previous_volume_position == "Top" and current_volume_position == "Bottom":
#         previous_volume_position = current_volume_position
#         volume_change_time = time.time()
#         return Action.VolumeDown
#       elif previous_volume_position == "Bottom" and current_volume_position == "Top":
#         previous_volume_position = current_volume_position
#         volume_change_time = time.time()
#         return Action.VolumeUp
            
#   previous_volume_position = current_volume_position
#   volume_change_time = time.time()
#   return Action.VolumeControl

def controlVolume(y_list, screen_height, isFingersInCircle):
    global previous_volume_position, current_volume_position, volume_change_time

    if isFingersInCircle[1] and isFingersInCircle[2]:
        return Action.Nothing, None

    thumb_tip_y = y_list[mp.solutions.hands.HandLandmark.THUMB_TIP.value]

    thumb_y_percent = (screen_height - thumb_tip_y) / screen_height * 100

    return Action.VolumeControl, thumb_y_percent



def controlBackAndFront(isFingersInCircle, x, width):
  global previous_position, current_position, move_change_time
  # if any(isFingersInCircle):
  #   previous_position = None
  #   current_position = None
  #   return Action.Nothing
  # 왼손 위치
  left_bound = width // 3
  right_bound = 2 * (width // 3)
  if x < left_bound:
    current_position = "Left"
  elif x > right_bound:
    current_position = "Right"
  # 이동 방향 감지
  if current_position != previous_position:
    if time.time() - move_change_time < 1: 
      if previous_position == "Left" and current_position == "Right":
        previous_position = current_position
        move_change_time = time.time() 
        return Action.Back
      elif previous_position == "Right" and current_position == "Left":
        previous_position = current_position
        move_change_time = time.time() 
        return Action.Front
  previous_position = current_position
  move_change_time = time.time() 
  return Action.BnFReady

def controlMute():
  global isMute, last_mute_action_time
  if time.time() - last_mute_action_time > 1:
    last_mute_action_time = time.time()
    return Action.Mute
  else:
    return Action.Nothing

# 변수 초기화 (Initialize variables)
current_right_mode = RightMode.Nothing  # 현재 상태
previous_right_mode = RightMode.Nothing  # 이전 상태
right_change_time = time.time() 

current_left_mode = LeftMode.Nothing  # 현재 상태
previous_left_mode = LeftMode.Nothing  # 이전 상태
left_change_time = time.time()  # 상태 변경 시각

# for zoom action
zoom_change_time = time.time()
isZoom = False

# for back and front action
current_position = None
previous_position = None
move_change_time = time.time() 

# for mute action
last_mute_action_time = 0 

screen_x = 0
screen_y = 0

# for volumn
# previous_volume_position = None
# current_volume_position = None
# volume_change_time = 0 
volumePercent = 0

last_pointer_action_time = 0

# for scroll
last_scroll_action_time = 0
last_down_scroll_action_time = 0

STATE_HOLD_DURATION = 1  # 1초 동안 상태 유지 (Hold duration for state transition)

def main():
  global volumePercent
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_hands = mp.solutions.hands

  # For webcam input:
  cap = cv2.VideoCapture(0)
  with mp_hands.Hands(
      model_complexity=0,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue

      image = cv2.flip(image, 1)

      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)

      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
          height, width, _ = image.shape
          
          # 각 좌표 계산 (Calculate x, y coordinates)
          x_list = [int(landmark.x * width) for landmark in hand_landmarks.landmark]
          y_list = [int(landmark.y * height) for landmark in hand_landmarks.landmark]

          # 손 좌표를 화면 좌표로 변환
          global screen_x, screen_y
          index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
          screen_x = int(index_tip.x * screen_width)
          screen_y = int(index_tip.y * screen_height)

          # 손바닥 범위 계산 (Palm bounding range)
          rect_range = rectRangeOfPoints(x_list, y_list, [0, 1, 2, 5, 9, 10, 13, 14, 17, 18])
          circle_point = circleRangeOfPoints(rect_range)

          cv2.rectangle(image, rect_range[0], rect_range[1], (0, 255, 0), 3)
          cv2.circle(image, circle_point[0], circle_point[1], (0, 255, 0), 3)

          # 손가락 내부 범위 확인 (Check finger ranges)
          isFingersInCircle = []
          for i in range(5):
            isInCircle = isInCircleRange(circle_point, [(x_list[4*(i+1)], y_list[4*(i+1)])])
            isFingersInCircle.append(isInCircle)

          # 제스처 인식 및 동작 실행 (Gesture recognition and action execution)      
          handedness = results.multi_handedness[idx].classification[0]
          hand_label = handedness.label

          action = Action.Nothing

          if hand_label == "Left":
            new_mode = controlLeftMode(isFingersInCircle)
            leftStableModeState(new_mode)

            if current_left_mode == LeftMode.Volume:
              action, volumePercent = controlVolume(y_list, screen_height, isFingersInCircle)
            elif current_left_mode == LeftMode.BnFReady:
              action = controlBackAndFront(isFingersInCircle, x_list[0], width)
            elif current_left_mode == LeftMode.Mute:
              action = controlMute()
            cv2.putText(image, f"{action}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

          elif hand_label == "Right":
            new_mode = controlRightMode(isFingersInCircle)
            rightStableModeState(new_mode)

            if current_right_mode == RightMode.Pointer:
              action = controlPointer(isFingersInCircle)
            elif current_right_mode == RightMode.UpScroll:
              action = controlScroll()
            elif current_right_mode == RightMode.DownScroll:
              action = controlDownScroll()
            elif current_right_mode == RightMode.Zoom:
              action = controlZoom(isFingersInCircle, action)
            cv2.putText(image, f"{action}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

          # 동작에 따라 함수 실행
          if action in action_to_function:
            try:
              action_to_function[action]()
            except Exception as e:
              print(f"Error executing action {action}: {e}")

          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
      cv2.imshow('MediaPipe Hands', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()

if __name__ == "__main__":
    main()