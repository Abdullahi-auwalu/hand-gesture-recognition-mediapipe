#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import csv
import copy
import argparse
import itertools
from collections import Counter, deque
import math, time, uuid
from datetime import datetime

import cv2 as cv
import numpy as np
import mediapipe as mp

# ---- optional imports
try:
    import pyautogui
    HAVE_PYAUTO = True
    pyautogui.FAILSAFE = False
except Exception:
    HAVE_PYAUTO = False

try:
    import pyttsx3
    HAVE_TTS = True
except Exception:
    HAVE_TTS = False

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier

# ======================= CLI =======================
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--use_static_image_mode", action="store_true")
    p.add_argument("--min_detection_confidence", type=float, default=0.7)
    p.add_argument("--min_tracking_confidence", type=float, default=0.5)

    # virtual mouse
    p.add_argument("--virtual_mouse", action="store_true")
    # speech
    p.add_argument("--speak", action="store_true",
                   help="Say the current gesture aloud")
    p.add_argument("--speech_rate", type=int, default=175)
    p.add_argument("--speech_cooldown", type=float, default=0.8,
                   help="seconds between announcements")
    # recording
    p.add_argument("--recordings_dir", type=str, default="recordings")
    p.add_argument("--logs_dir", type=str, default="logs")
    p.add_argument("--config", type=str, default="config/gesture_bindings.json")
    return p.parse_args()

# ======================= math helpers =======================
def _dist(p1, p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
def _bbox_diag(b):  # [x1,y1,x2,y2]
    w = max(1, b[2]-b[0]); h = max(1, b[3]-b[1]); return math.hypot(w,h)
def _clamp(v, lo, hi): return max(lo, min(hi, v))
def _smooth(prev, new, alpha=0.35):
    if prev is None: return new
    return (1-alpha)*np.array(prev)+alpha*np.array(new)

# ======================= bindings =======================
def load_bindings(path, labels):
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {"virtual_mouse": {"enabled": True, "pointer_pose_label": "Pointer"}}
    label_to_id = {n:i for i,n in enumerate(labels)}
    pointer_label = (cfg.get("virtual_mouse") or {}).get("pointer_pose_label","Pointer")
    pointer_id = label_to_id.get(pointer_label, 2)
    return {
        "raw": cfg,
        "virtual_mouse": {"enabled": bool((cfg.get("virtual_mouse") or {}).get("enabled", True)),
                          "pointer_label": pointer_label, "pointer_id": pointer_id}
    }

# ======================= logger =======================
class RunLogger:
    def __init__(self, logs_dir, meta):
        os.makedirs(logs_dir, exist_ok=True)
        self.session = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(logs_dir, f"run_{self.session}.jsonl")
        self.fh = open(self.path, "a", encoding="utf-8")
        self.frame_idx = 0
        self._w({"type":"meta", **meta})
    def _w(self, obj):
        self.fh.write(json.dumps(obj, ensure_ascii=False)+"\n"); self.fh.flush()
    def log_frame(self, **kw):
        self.frame_idx += 1
        self._w({"type":"frame", "ts":time.time(), "frame_idx":self.frame_idx, **kw})
    def log_event(self, event, **kw):
        self._w({"type":"event","ts":time.time(),"event":event,**kw})
    def close(self):
        try: self.fh.close()
        except Exception: pass

# ======================= Video recorder =======================
class VideoRecorder:
    def __init__(self, out_dir, w, h, fps=30):
        self.dir = out_dir; os.makedirs(self.dir, exist_ok=True)
        self.w, self.h, self.fps = w, h, fps
        self.writer = None; self.filepath = None; self.started_at = None
        self.fourcc = cv.VideoWriter_fourcc(*"mp4v")
    def is_on(self): return self.writer is not None
    def start(self):
        if self.is_on(): return self.filepath
        fname = f"rec_{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"
        self.filepath = os.path.join(self.dir, fname)
        self.writer = cv.VideoWriter(self.filepath, self.fourcc, self.fps, (self.w,self.h))
        self.started_at = time.time()
        return self.filepath
    def write(self, frame):
        if self.writer is not None: self.writer.write(frame)
    def stop(self):
        if self.writer is None: return None
        self.writer.release(); self.writer = None
        dur = time.time()-self.started_at if self.started_at else None
        fp = self.filepath
        self.filepath = None; self.started_at = None
        return fp, dur

# ======================= Virtual mouse (unchanged behavior) =======================
class VirtualMouse:
    def __init__(self, frame_w, frame_h, pointer_id=2, enable_now=False):
        self.enabled = enable_now and HAVE_PYAUTO
        self.frame_w, self.frame_h = frame_w, frame_h
        self.pointer_id = pointer_id
        if HAVE_PYAUTO: self.screen_w, self.screen_h = pyautogui.size()
        else: self.screen_w = self.screen_h = 0
        self.cursor_prev = None; self.last_click_time = 0.0; self.last_scroll_time = 0.0
        self.stable_counter = 0; self.STABLE_N = 4
        self.tip_trail = deque(maxlen=24)
        self.CONF_THRESHOLD = 0.60; self.PINCH_FRAC = 0.12
        self.CLICK_COOLDOWN = 0.30; self.SCROLL_COOLDOWN = 0.25
        self.CIRCLE_MIN_AREA = 1200.0; self.CIRCLE_MIN_POINTS = 12
        self.SMOOTH_ALPHA = 0.35; self.MARGIN = 0.05
        self.pinch_start_time = None; self._pinch_debug_ratio=None
    def set_pointer_id(self, pid): self.pointer_id=int(pid)
    def toggle(self): self.enabled = not self.enabled and HAVE_PYAUTO; return self.enabled
    def map_to_screen(self, ix, iy):
        sx = (ix/max(1,self.frame_w))*self.screen_w
        sy = (iy/max(1,self.frame_h))*self.screen_h
        sx = _clamp(sx, self.screen_w*0.05, self.screen_w*0.95)
        sy = _clamp(sy, self.screen_h*0.05, self.screen_h*0.95)
        return int(sx), int(sy)
    def stable_gate(self, is_ptr, conf_ok):
        self.stable_counter = min(self.STABLE_N, self.stable_counter+1) if (is_ptr and conf_ok) else 0
        return self.stable_counter >= self.STABLE_N
    def _is_pinching(self, thumb_tip, index_tip, brect):
        diag = _bbox_diag(brect); 
        if diag < 1: self._pinch_debug_ratio=None; return False
        ratio = _dist(thumb_tip,index_tip)/diag; self._pinch_debug_ratio=ratio
        return ratio < self.PINCH_FRAC
    def handle_pinch_left_click(self, thumb_tip, index_tip, brect):
        if not HAVE_PYAUTO: return False
        now=time.time(); pinching=self._is_pinching(thumb_tip,index_tip,brect)
        if pinching:
            if self.pinch_start_time is None: self.pinch_start_time = now
            return False
        else:
            if self.pinch_start_time is not None and (now-self.last_click_time)>=self.CLICK_COOLDOWN:
                try: pyautogui.click(button='left')
                except Exception: pass
                self.last_click_time = now; self.pinch_start_time=None; return True
            self.pinch_start_time=None; return False
    def _signed_area(self, pts):
        if len(pts)<3: return 0.0
        a=0.0
        for i in range(len(pts)):
            x1,y1=pts[i]; x2,y2=pts[(i+1)%len(pts)]; a+=(x1*y2-x2*y1)
        return 0.5*a
    def maybe_scroll(self):
        if not HAVE_PYAUTO or len(self.tip_trail)<self.CIRCLE_MIN_POINTS: return False
        now=time.time()
        if now-self.last_scroll_time<self.SCROLL_COOLDOWN: return False
        area=self._signed_area(list(self.tip_trail))
        if abs(area)>self.CIRCLE_MIN_AREA:
            delta=-1 if area>0 else 1
            try: pyautogui.scroll(120*delta)
            except Exception: pass
            self.last_scroll_time=now; return True
        return False
    def update(self, brect, lmk_px, hand_sign_id, hand_conf):
        out={"pinch_ratio":None,"clicked":False,"scrolled":False}
        if not self.enabled:
            self.cursor_prev=None; self.tip_trail.clear(); self.pinch_start_time=None; self._pinch_debug_ratio=None
            return out
        conf_ok = hand_conf>=self.CONF_THRESHOLD
        is_pointer = (hand_sign_id==self.pointer_id)
        index_tip=lmk_px[8]; thumb_tip=lmk_px[4]
        if self.stable_gate(is_pointer, conf_ok):
            sx,sy=self.map_to_screen(index_tip[0], index_tip[1])
            self.cursor_prev=_smooth(self.cursor_prev,(sx,sy),alpha=0.35)
            cx,cy=int(self.cursor_prev[0]), int(self.cursor_prev[1])
            if HAVE_PYAUTO:
                try: pyautogui.moveTo(cx,cy, duration=0)
                except Exception: pass
            self.tip_trail.append((index_tip[0],index_tip[1]))
            if self.maybe_scroll(): out["scrolled"]=True
        else:
            self.cursor_prev=None; self.tip_trail.clear()
        if self.handle_pinch_left_click(thumb_tip,index_tip,brect): out["clicked"]=True
        out["pinch_ratio"]=self._pinch_debug_ratio; return out

# ======================= main =======================
def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.config), exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.recordings_dir, exist_ok=True)

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # estimate fps for recording; default 30
    fps_for_rec = cap.get(cv.CAP_PROP_FPS) or 30

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    CONF_THRESHOLD = 0.60

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_labels = [row[0] for row in csv.reader(f)]

    bindings = load_bindings(args.config, keypoint_labels)
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    mode = 0
    session_id = str(uuid.uuid4())

    vmouse = VirtualMouse(args.width, args.height,
                          pointer_id=bindings["virtual_mouse"]["pointer_id"],
                          enable_now=(args.virtual_mouse and bindings["virtual_mouse"]["enabled"]))
    if args.virtual_mouse and not HAVE_PYAUTO:
        print("[virtual-mouse] pyautogui not available. pip install pyautogui")

    # ---- TTS
    tts_engine = None
    speak_enabled = bool(args.speak and HAVE_TTS)
    last_spoken = ""
    last_speak_ts = 0.0
    if speak_enabled:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", args.speech_rate)

    # ---- Recorder
    recorder = VideoRecorder(args.recordings_dir, args.width, args.height, fps=fps_for_rec)

    meta = {
        "session_id": session_id,
        "labels": keypoint_labels,
        "pointer_label": bindings["virtual_mouse"]["pointer_label"],
        "pointer_id": bindings["virtual_mouse"]["pointer_id"],
        "virtual_mouse_enabled_at_start": vmouse.enabled,
        "args": vars(args),
    }
    logger = RunLogger(args.logs_dir, meta)

    try:
        while True:
            fps = cvFpsCalc.get()
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            if key in (ord('m'), ord('M')):
                now_enabled = vmouse.toggle()
                logger.log_event("vmouse_toggle", enabled=now_enabled)
                print(f"[virtual-mouse] {'ENABLED' if now_enabled else 'DISABLED'}")
            if key in (ord('v'), ord('V')) and HAVE_TTS:
                speak_enabled = not speak_enabled
                last_spoken = ""
                logger.log_event("speech_toggle", enabled=speak_enabled)
                print(f"[speech] {'ENABLED' if speak_enabled else 'DISABLED'}")
            if key in (ord('r'), ord('R')):
                if recorder.is_on():
                    fp, dur = recorder.stop()
                    logger.log_event("record_stop", file=os.path.basename(fp), duration_s=dur)
                    print(f"[rec] stopped → {fp} ({dur:.1f}s)")
                else:
                    fp = recorder.start()
                    logger.log_event("record_start", file=os.path.basename(fp))
                    print(f"[rec] started → {fp}")

            number, mode = select_mode(key, mode)

            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)

            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            log_record = {"fps": fps, "vmouse_enabled": vmouse.enabled, "hand": None}

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    pre_lmk = pre_process_landmark(landmark_list)
                    pre_hist = pre_process_point_history(debug_image, point_history)
                    logging_csv(number, mode, pre_lmk, pre_hist)

                    try:
                        hand_sign_id, hand_conf, _ = keypoint_classifier.predict(pre_lmk)
                    except AttributeError:
                        idx = keypoint_classifier(pre_lmk); hand_sign_id=int(idx); hand_conf=1.0

                    if hand_sign_id == bindings["virtual_mouse"]["pointer_id"]:
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    finger_gesture_id = 0
                    if len(pre_hist) == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_hist)
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg = Counter(finger_gesture_history).most_common(1)[0][0]

                    # Virtual mouse
                    vm_info = vmouse.update(brect, landmark_list, hand_sign_id, hand_conf)

                    # Draw & label
                    label_text = ("Unknown" if hand_conf < CONF_THRESHOLD else keypoint_labels[hand_sign_id])
                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(debug_image, brect, handedness,
                                                 label_text, point_history_labels[most_common_fg])

                    # Speak (rate-limited & only on change)
                    now = time.time()
                    if speak_enabled and hand_conf >= CONF_THRESHOLD and label_text != last_spoken and (now - last_speak_ts) >= args.speech_cooldown:
                        try:
                            tts_engine.say(label_text); tts_engine.iterate()  # non-blocking-ish
                        except Exception:
                            pass
                        last_spoken = label_text; last_speak_ts = now

                    # Log one hand (latest)
                    log_record["hand"] = {
                        "label": label_text, "id": int(hand_sign_id), "conf": float(hand_conf),
                        "pinch_ratio": (vmouse._pinch_debug_ratio if vmouse._pinch_debug_ratio is not None else None),
                        "clicked": bool(vm_info["clicked"]), "scrolled": bool(vm_info["scrolled"]),
                        "bbox": list(map(int, brect)), "handedness": handedness.classification[0].label
                    }
            else:
                point_history.append([0, 0])

            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, fps, mode, number)

            # HUD lines
            cv.putText(debug_image, f"VMOUSE: {'ON' if vmouse.enabled else 'OFF'} (m)",
                       (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4, cv.LINE_AA)
            cv.putText(debug_image, f"VMOUSE: {'ON' if vmouse.enabled else 'OFF'} (m)",
                       (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)

            cv.putText(debug_image, f"SPEECH: {'ON' if speak_enabled else 'OFF'} (v)",
                       (10, 175), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4, cv.LINE_AA)
            cv.putText(debug_image, f"SPEECH: {'ON' if speak_enabled else 'OFF'} (v)",
                       (10, 175), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)

            rec_txt = f"REC: {'ON' if recorder.is_on() else 'OFF'} (r)"
            cv.putText(debug_image, rec_txt, (10, 200), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4, cv.LINE_AA)
            cv.putText(debug_image, rec_txt, (10, 200), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if recorder.is_on() else (255,255,255), 1, cv.LINE_AA)

            # write video if recording
            if recorder.is_on():
                recorder.write(image)  # raw (not debug overlay) to file

            cv.imshow("Hand Gesture Recognition", debug_image)

            loop_ms = (time.time() - last_speak_ts) * 1000.0  # harmless reuse
            logger.log_frame(latency_ms=loop_ms, **log_record)
    finally:
        if recorder.is_on():
            recorder.stop()
        if HAVE_TTS and tts_engine:
            try: tts_engine.stop()
            except Exception: pass
        logger.close()
        cap.release(); cv.destroyAllWindows()

# -------- helpers (unchanged drawing / calc) --------
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57: number = key - 48
    if key == 110: mode = 0  # n
    if key == 107: mode = 1  # k
    if key == 104: mode = 2  # h
    return number, mode

def calc_bounding_rect(image, landmarks):
    w,h = image.shape[1], image.shape[0]
    arr = np.empty((0,2), int)
    for lm in landmarks.landmark:
        x = min(int(lm.x * w), w-1); y = min(int(lm.y * h), h-1)
        arr = np.append(arr, [np.array((x,y))], axis=0)
    x,y,ww,hh = cv.boundingRect(arr); return [x,y,x+ww,y+hh]

def calc_landmark_list(image, landmarks):
    w,h = image.shape[1], image.shape[0]
    pts=[]
    for lm in landmarks.landmark:
        x = min(int(lm.x * w), w-1); y = min(int(lm.y * h), h-1); pts.append([x,y])
    return pts

def pre_process_landmark(landmark_list):
    tmp = copy.deepcopy(landmark_list); base_x,base_y=0,0
    for i,pt in enumerate(tmp):
        if i==0: base_x,base_y=pt[0],pt[1]
        tmp[i][0]-=base_x; tmp[i][1]-=base_y
    tmp = list(itertools.chain.from_iterable(tmp))
    mv = max(list(map(abs,tmp))) if tmp else 1.0
    return [0.0 for _ in tmp] if mv==0 else [v/mv for v in tmp]

def pre_process_point_history(image, point_history):
    w,h = image.shape[1], image.shape[0]
    tmp = copy.deepcopy(point_history); base_x,base_y=0,0
    for i,p in enumerate(tmp):
        if i==0: base_x,base_y=p[0],p[1]
        tmp[i][0]=(tmp[i][0]-base_x)/w; tmp[i][1]=(tmp[i][1]-base_y)/h
    return list(itertools.chain.from_iterable(tmp))

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0: return
    if mode == 1 and (0 <= number <= 9):
        with open('model/keypoint_classifier/keypoint.csv','a',newline="") as f:
            csv.writer(f).writerow([number,*landmark_list])
    if mode == 2 and (0 <= number <= 9):
        with open('model/point_history_classifier/point_history.csv','a',newline="") as f:
            csv.writer(f).writerow([number,*point_history_list])

def draw_landmarks(image, lmk):
    if len(lmk)>0:
        # fingers + palm (same as before)
        for a,b in [(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),
                    (13,14),(14,15),(15,16),(17,18),(18,19),(19,20),
                    (0,1),(1,2),(2,5),(5,9),(9,13),(13,17),(17,0)]:
            cv.line(image, tuple(lmk[a]), tuple(lmk[b]), (0,0,0), 6)
            cv.line(image, tuple(lmk[a]), tuple(lmk[b]), (255,255,255), 2)
    for i,(x,y) in enumerate(lmk):
        r = 8 if i in (4,8,12,16,20) else 5
        cv.circle(image,(x,y),r,(255,255,255),-1); cv.circle(image,(x,y),r,(0,0,0),1)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect: cv.rectangle(image,(brect[0],brect[1]),(brect[2],brect[3]),(0,0,0),1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image,(brect[0],brect[1]),(brect[2],brect[1]-22),(0,0,0),-1)
    info = handedness.classification[0].label
    if hand_sign_text: info = f"{info}:{hand_sign_text}"
    cv.putText(image, info, (brect[0]+5,brect[1]-4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1,cv.LINE_AA)
    return image

def draw_point_history(image, ph):
    for i,p in enumerate(ph):
        if p[0]!=0 and p[1]!=0:
            cv.circle(image,(p[0],p[1]),1+int(i/2),(152,251,152),2)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:"+str(fps), (10,30), cv.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),4,cv.LINE_AA)
    cv.putText(image, "FPS:"+str(fps), (10,30), cv.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2,cv.LINE_AA)
    mode_str=['Logging Key Point','Logging Point History']
    if 1<=mode<=2:
        cv.putText(image, "MODE:"+mode_str[mode-1], (10,90), cv.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv.LINE_AA)
        if 0<=number<=9:
            cv.putText(image, "NUM:"+str(number), (10,110), cv.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv.LINE_AA)
    return image

if __name__ == "__main__":
    main()



# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import os
# import json
# import csv
# import copy
# import argparse
# import itertools
# from collections import Counter, deque
# import math
# import time
# from datetime import datetime
# import uuid

# import cv2 as cv
# import numpy as np
# import mediapipe as mp

# # --- optional import guard for pyautogui ------------------------------------
# try:
#     import pyautogui
#     HAVE_PYAUTO = True
#     pyautogui.FAILSAFE = False
# except Exception:
#     HAVE_PYAUTO = False
# # ----------------------------------------------------------------------------

# from utils import CvFpsCalc
# from model import KeyPointClassifier
# from model import PointHistoryClassifier


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--device", type=int, default=0)
#     parser.add_argument("--width", type=int, default=960, help="cap width")
#     parser.add_argument("--height", type=int, default=540, help="cap height")
#     parser.add_argument("--use_static_image_mode", action="store_true")
#     parser.add_argument("--min_detection_confidence", type=float, default=0.7)
#     parser.add_argument("--min_tracking_confidence", type=float, default=0.5)

#     # virtual mouse toggle
#     parser.add_argument("--virtual_mouse", action="store_true",
#                         help="Enable virtual mouse (pointer mode)")
#     # where logs/config go
#     parser.add_argument("--logs_dir", type=str, default="logs",
#                         help="Folder for JSONL run logs")
#     parser.add_argument("--config", type=str,
#                         default="config/gesture_bindings.json",
#                         help="Gesture→action bindings")
#     return parser.parse_args()


# # ---------- Utility math helpers -------------------------------------------
# def _dist(p1, p2):
#     return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# def _bbox_diag(brect):
#     w = max(1, brect[2] - brect[0])
#     h = max(1, brect[3] - brect[1])
#     return math.hypot(w, h)


# def _clamp(v, lo, hi):
#     return max(lo, min(hi, v))


# def _smooth(prev, new, alpha=0.35):
#     if prev is None:
#         return new
#     return (1 - alpha) * np.array(prev) + alpha * np.array(new)
# # ---------------------------------------------------------------------------


# # ---------- bindings loader -------------------------------------------------
# def load_bindings(path, labels):
#     """
#     Load gesture→action bindings and resolve label names to class indices.
#     Returns a dict with resolved ids (e.g., virtual_mouse.pointer_id).
#     """
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             cfg = json.load(f)
#     except Exception:
#         # sensible defaults if config missing
#         cfg = {
#             "virtual_mouse": {
#                 "enabled": True,
#                 "pointer_pose_label": "Pointer"   # must exist in labels CSV
#             },
#             "click": {"mode": "pinch"},          # pinch-only in our app
#         }

#     label_to_id = {name: i for i, name in enumerate(labels)}

#     # virtual mouse
#     vm_cfg = cfg.get("virtual_mouse", {})
#     pointer_label = vm_cfg.get("pointer_pose_label", "Pointer")
#     pointer_id = label_to_id.get(pointer_label, 2)  # fallback to id=2

#     resolved = {
#         "raw": cfg,
#         "virtual_mouse": {
#             "enabled": bool(vm_cfg.get("enabled", True)),
#             "pointer_label": pointer_label,
#             "pointer_id": pointer_id,
#         },
#         "click": cfg.get("click", {"mode": "pinch"})
#     }
#     return resolved
# # ---------------------------------------------------------------------------


# # ---------- JSONL run logger -----------------------------------------------
# class RunLogger:
#     """
#     Append one JSON object per frame to logs/<session>.jsonl
#     """
#     def __init__(self, logs_dir, meta):
#         os.makedirs(logs_dir, exist_ok=True)
#         session = datetime.now().strftime("%Y%m%d-%H%M%S")
#         self.path = os.path.join(logs_dir, f"run_{session}.jsonl")
#         self.fh = open(self.path, "a", encoding="utf-8")
#         self.meta = meta
#         self.frame_idx = 0
#         self._write({"type": "meta", **meta})

#     def _write(self, obj):
#         self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
#         self.fh.flush()

#     def log_frame(self, **kwargs):
#         self.frame_idx += 1
#         rec = {
#             "type": "frame",
#             "ts": time.time(),
#             "frame_idx": self.frame_idx,
#             **kwargs
#         }
#         self._write(rec)

#     def log_event(self, event, **kwargs):
#         rec = {"type": "event", "ts": time.time(), "event": event, **kwargs}
#         self._write(rec)

#     def close(self):
#         try:
#             self.fh.close()
#         except Exception:
#             pass
# # ---------------------------------------------------------------------------


# # ---------- Virtual mouse logic --------------------------------------------
# class VirtualMouse:
#     """
#     Cursor moves with index fingertip (landmark 8).
#     - Movement is gated by a stable 'Pointer' pose (id from bindings).
#     - Pinch (index tip–thumb tip close) -> Left click (fires on release).
#     - Circle loop (rough) -> Scroll.
#     """
#     def __init__(self, frame_w, frame_h, pointer_id=2, enable_now=False):
#         self.enabled = enable_now and HAVE_PYAUTO
#         self.frame_w = frame_w
#         self.frame_h = frame_h
#         self.pointer_id = pointer_id

#         if HAVE_PYAUTO:
#             self.screen_w, self.screen_h = pyautogui.size()
#         else:
#             self.screen_w = self.screen_h = 0

#         # state
#         self.cursor_prev = None
#         self.last_click_time = 0.0
#         self.last_scroll_time = 0.0

#         # stability gate
#         self.stable_counter = 0
#         self.STABLE_N = 4

#         # trails / thresholds
#         self.tip_trail = deque(maxlen=24)
#         self.CONF_THRESHOLD = 0.60
#         self.PINCH_FRAC = 0.12
#         self.CLICK_COOLDOWN = 0.30
#         self.SCROLL_COOLDOWN = 0.25
#         self.CIRCLE_MIN_AREA = 1200.0
#         self.CIRCLE_MIN_POINTS = 12

#         # mapping controls
#         self.SMOOTH_ALPHA = 0.35
#         self.MARGIN = 0.05

#         # pinch state + HUD
#         self.pinch_start_time = None
#         self._pinch_debug_ratio = None

#     def set_pointer_id(self, pid):
#         self.pointer_id = int(pid)

#     def toggle(self):
#         self.enabled = not self.enabled and HAVE_PYAUTO
#         return self.enabled

#     def map_to_screen(self, ix, iy):
#         sx = (ix / max(1, self.frame_w)) * self.screen_w
#         sy = (iy / max(1, self.frame_h)) * self.screen_h
#         sx = _clamp(sx, self.screen_w * self.MARGIN, self.screen_w * (1 - self.MARGIN))
#         sy = _clamp(sy, self.screen_h * self.MARGIN, self.screen_h * (1 - self.MARGIN))
#         return int(sx), int(sy)

#     def stable_gate(self, is_pointer, conf_ok):
#         if is_pointer and conf_ok:
#             self.stable_counter = min(self.STABLE_N, self.stable_counter + 1)
#         else:
#             self.stable_counter = 0
#         return self.stable_counter >= self.STABLE_N

#     def _is_pinching(self, thumb_tip, index_tip, brect):
#         diag = _bbox_diag(brect)
#         if diag < 1:
#             self._pinch_debug_ratio = None
#             return False
#         ratio = _dist(thumb_tip, index_tip) / diag
#         self._pinch_debug_ratio = ratio
#         return ratio < self.PINCH_FRAC

#     def handle_pinch_left_click(self, thumb_tip, index_tip, brect):
#         """
#         Returns True if a click was fired this frame.
#         """
#         if not HAVE_PYAUTO:
#             return False
#         now = time.time()
#         pinching = self._is_pinching(thumb_tip, index_tip, brect)

#         if pinching:
#             if self.pinch_start_time is None:
#                 self.pinch_start_time = now
#             return False
#         else:
#             # released → click once (debounced)
#             if self.pinch_start_time is not None:
#                 if (now - self.last_click_time) >= self.CLICK_COOLDOWN:
#                     try:
#                         pyautogui.click(button='left')
#                     except Exception:
#                         pass
#                     self.last_click_time = now
#                     self.pinch_start_time = None
#                     return True
#             self.pinch_start_time = None
#             return False

#     def _signed_area(self, pts):
#         if len(pts) < 3:
#             return 0.0
#         area = 0.0
#         for i in range(len(pts)):
#             x1, y1 = pts[i]
#             x2, y2 = pts[(i + 1) % len(pts)]
#             area += (x1 * y2 - x2 * y1)
#         return 0.5 * area

#     def maybe_scroll(self):
#         if not HAVE_PYAUTO:
#             return False
#         if len(self.tip_trail) < self.CIRCLE_MIN_POINTS:
#             return False
#         now = time.time()
#         if now - self.last_scroll_time < self.SCROLL_COOLDOWN:
#             return False
#         area = self._signed_area(list(self.tip_trail))
#         if abs(area) > self.CIRCLE_MIN_AREA:
#             delta = -1 if area > 0 else 1
#             try:
#                 pyautogui.scroll(120 * delta)
#             except Exception:
#                 pass
#             self.last_scroll_time = now
#             return True
#         return False

#     def update(self, brect, landmarks_px, hand_sign_id, hand_conf):
#         """
#         Returns a dict with HUD/debug values and event flags:
#         {"pinch_ratio": float|None, "clicked": bool, "scrolled": bool}
#         """
#         out = {"pinch_ratio": None, "clicked": False, "scrolled": False}

#         if not self.enabled:
#             self.cursor_prev = None
#             self.tip_trail.clear()
#             self.pinch_start_time = None
#             self._pinch_debug_ratio = None
#             return out

#         conf_ok = hand_conf >= self.CONF_THRESHOLD
#         is_pointer = (hand_sign_id == self.pointer_id)

#         index_tip = landmarks_px[8]
#         thumb_tip = landmarks_px[4]

#         # move when stable pointer
#         if self.stable_gate(is_pointer, conf_ok):
#             sx, sy = self.map_to_screen(index_tip[0], index_tip[1])
#             self.cursor_prev = _smooth(self.cursor_prev, (sx, sy), alpha=self.SMOOTH_ALPHA)
#             cx, cy = int(self.cursor_prev[0]), int(self.cursor_prev[1])
#             if HAVE_PYAUTO:
#                 try:
#                     pyautogui.moveTo(cx, cy, duration=0)
#                 except Exception:
#                     pass
#             self.tip_trail.append((index_tip[0], index_tip[1]))
#             if self.maybe_scroll():
#                 out["scrolled"] = True
#         else:
#             self.cursor_prev = None
#             self.tip_trail.clear()

#         # pinch click (edge-triggered)
#         if self.handle_pinch_left_click(thumb_tip, index_tip, brect):
#             out["clicked"] = True

#         out["pinch_ratio"] = self._pinch_debug_ratio
#         return out
# # ---------------------------------------------------------------------------


# def main():
#     args = get_args()

#     # dirs
#     os.makedirs(os.path.dirname(args.config), exist_ok=True)
#     os.makedirs(args.logs_dir, exist_ok=True)

#     cap = cv.VideoCapture(args.device)
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(
#         static_image_mode=args.use_static_image_mode,
#         max_num_hands=2,
#         min_detection_confidence=args.min_detection_confidence,
#         min_tracking_confidence=args.min_tracking_confidence,
#     )

#     keypoint_classifier = KeyPointClassifier()
#     point_history_classifier = PointHistoryClassifier()

#     # Confidence threshold for label display
#     CONF_THRESHOLD = 0.60

#     # Labels
#     with open('model/keypoint_classifier/keypoint_classifier_label.csv',
#               encoding='utf-8-sig') as f:
#         keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
#     with open('model/point_history_classifier/point_history_classifier_label.csv',
#               encoding='utf-8-sig') as f:
#         point_history_classifier_labels = [row[0] for row in csv.reader(f)]

#     # Bindings (uses label names → resolves to ids)
#     bindings = load_bindings(args.config, keypoint_classifier_labels)

#     cvFpsCalc = CvFpsCalc(buffer_len=10)

#     history_length = 16
#     point_history = deque(maxlen=history_length)
#     finger_gesture_history = deque(maxlen=history_length)

#     mode = 0
#     session_id = str(uuid.uuid4())

#     # virtual mouse instance
#     vmouse = VirtualMouse(
#         args.width, args.height,
#         pointer_id=bindings["virtual_mouse"]["pointer_id"],
#         enable_now=(args.virtual_mouse and bindings["virtual_mouse"]["enabled"])
#     )
#     if args.virtual_mouse and not HAVE_PYAUTO:
#         print("[virtual-mouse] pyautogui not available. Install with: pip install pyautogui")

#     # run logger
#     meta = {
#         "session_id": session_id,
#         "labels": keypoint_classifier_labels,
#         "pointer_label": bindings["virtual_mouse"]["pointer_label"],
#         "pointer_id": bindings["virtual_mouse"]["pointer_id"],
#         "virtual_mouse_enabled_at_start": vmouse.enabled,
#         "args": vars(args)
#     }
#     logger = RunLogger(args.logs_dir, meta)

#     try:
#         while True:
#             loop_t0 = time.time()
#             fps = cvFpsCalc.get()

#             key = cv.waitKey(10)
#             if key == 27:  # ESC
#                 break

#             # toggle virtual mouse with 'm'
#             if key in (ord('m'), ord('M')):
#                 now_enabled = vmouse.toggle()
#                 logger.log_event("vmouse_toggle", enabled=now_enabled)
#                 print(f"[virtual-mouse] {'ENABLED' if now_enabled else 'DISABLED'}")

#             number, mode = select_mode(key, mode)

#             ret, image = cap.read()
#             if not ret:
#                 break
#             image = cv.flip(image, 1)
#             debug_image = copy.deepcopy(image)

#             image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#             image_rgb.flags.writeable = False
#             results = hands.process(image_rgb)
#             image_rgb.flags.writeable = True

#             # Defaults for logging when no hands
#             log_record = {
#                 "fps": fps,
#                 "vmouse_enabled": vmouse.enabled,
#                 # top-level prediction fields (None if no hand)
#                 "pred_label": None,
#                 "pred_id": None,
#                 "pred_conf": None,
#                 "classifier_ms": None,
#                 "hand": None
#             }

#             if results.multi_hand_landmarks is not None:
#                 for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
#                                                       results.multi_handedness):
#                     brect = calc_bounding_rect(debug_image, hand_landmarks)
#                     landmark_list = calc_landmark_list(debug_image, hand_landmarks)

#                     pre_processed_landmark_list = pre_process_landmark(landmark_list)
#                     pre_processed_point_history_list = pre_process_point_history(
#                         debug_image, point_history)

#                     logging_csv(number, mode, pre_processed_landmark_list,
#                                 pre_processed_point_history_list)

#                     # classifier API (new or old) + measure latency
#                     cls_t0 = time.time()
#                     try:
#                         hand_sign_id, hand_conf, _probs = keypoint_classifier.predict(
#                             pre_processed_landmark_list
#                         )
#                     except AttributeError:
#                         idx = keypoint_classifier(pre_processed_landmark_list)
#                         hand_sign_id = int(idx)
#                         hand_conf = 1.0
#                         _probs = None
#                     classifier_ms = (time.time() - cls_t0) * 1000.0

#                     # pointer trail
#                     if hand_sign_id == bindings["virtual_mouse"]["pointer_id"]:
#                         point_history.append(landmark_list[8])
#                     else:
#                         point_history.append([0, 0])

#                     # finger gesture (history)
#                     finger_gesture_id = 0
#                     if len(pre_processed_point_history_list) == (history_length * 2):
#                         finger_gesture_id = point_history_classifier(
#                             pre_processed_point_history_list)

#                     finger_gesture_history.append(finger_gesture_id)
#                     most_common_fg_id = Counter(finger_gesture_history).most_common()
#                     most_common_fg = most_common_fg_id[0][0] if most_common_fg_id else 0

#                     # virtual mouse update
#                     vm_info = vmouse.update(brect, landmark_list, hand_sign_id, hand_conf)

#                     # draw
#                     hand_sign_text = ("Unknown" if hand_conf < CONF_THRESHOLD
#                                       else keypoint_classifier_labels[hand_sign_id])
#                     debug_image = draw_bounding_rect(True, debug_image, brect)
#                     debug_image = draw_landmarks(debug_image, landmark_list)
#                     debug_image = draw_info_text(
#                         debug_image,
#                         brect,
#                         handedness,
#                         hand_sign_text,
#                         point_history_classifier_labels[most_common_fg],
#                     )

#                     # set top-level prediction fields so the API/UI can read them easily
#                     log_record["pred_label"] = keypoint_classifier_labels[hand_sign_id]
#                     log_record["pred_id"] = int(hand_sign_id)
#                     log_record["pred_conf"] = float(hand_conf)
#                     log_record["classifier_ms"] = float(classifier_ms)

#                     # per-hand detail (kept for richness / debugging)
#                     log_record["hand"] = {
#                         "label": keypoint_classifier_labels[hand_sign_id],
#                         "id": int(hand_sign_id),
#                         "conf": float(hand_conf),
#                         "pinch_ratio": (None if vm_info["pinch_ratio"] is None
#                                         else float(vmouse._pinch_debug_ratio)),
#                         "clicked": bool(vm_info["clicked"]),
#                         "scrolled": bool(vm_info["scrolled"]),
#                         "bbox": list(map(int, brect)),
#                         "handedness": handedness.classification[0].label
#                     }
#             else:
#                 point_history.append([0, 0])

#             debug_image = draw_point_history(debug_image, point_history)
#             debug_image = draw_info(debug_image, fps, mode, number)

#             # HUD for virtual mouse + pinch ratio
#             hud1 = f"VMOUSE: {'ON' if vmouse.enabled else 'OFF'} (press 'm' to toggle)"
#             cv.putText(debug_image, hud1, (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6,
#                        (0, 0, 0), 4, cv.LINE_AA)
#             cv.putText(debug_image, hud1, (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6,
#                        (255, 255, 255), 1, cv.LINE_AA)

#             ratio = vmouse._pinch_debug_ratio
#             status = "PINCH" if vmouse.pinch_start_time is not None else "OPEN"
#             hud2 = (f"pinch_ratio: {ratio:.3f}  thr:{vmouse.PINCH_FRAC:.2f}  state:{status}"
#                     if ratio is not None else "pinch_ratio: n/a")
#             cv.putText(debug_image, hud2, (10, 175), cv.FONT_HERSHEY_SIMPLEX, 0.6,
#                        (0, 0, 0), 4, cv.LINE_AA)
#             cv.putText(debug_image, hud2, (10, 175), cv.FONT_HERSHEY_SIMPLEX, 0.6,
#                        (255, 255, 255), 1, cv.LINE_AA)

#             cv.imshow("Hand Gesture Recognition", debug_image)

#             # per-frame log write
#             loop_ms = (time.time() - loop_t0) * 1000.0
#             logger.log_frame(latency_ms=loop_ms, **log_record)

#         # end while
#     finally:
#         logger.close()
#         cap.release()
#         cv.destroyAllWindows()


# # ------------ drawing & helpers ----------------------
# def select_mode(key, mode):
#     number = -1
#     if 48 <= key <= 57:  # 0~9
#         number = key - 48
#     if key == 110:  # n
#         mode = 0
#     if key == 107:  # k
#         mode = 1
#     if key == 104:  # h
#         mode = 2
#     return number, mode


# def calc_bounding_rect(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]
#     landmark_array = np.empty((0, 2), int)
#     for _, lm in enumerate(landmarks.landmark):
#         x = min(int(lm.x * image_width), image_width - 1)
#         y = min(int(lm.y * image_height), image_height - 1)
#         landmark_array = np.append(landmark_array, [np.array((x, y))], axis=0)
#     x, y, w, h = cv.boundingRect(landmark_array)
#     return [x, y, x + w, y + h]


# def calc_landmark_list(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]
#     pts = []
#     for _, lm in enumerate(landmarks.landmark):
#         x = min(int(lm.x * image_width), image_width - 1)
#         y = min(int(lm.y * image_height), image_height - 1)
#         pts.append([x, y])
#     return pts


# def pre_process_landmark(landmark_list):
#     tmp = copy.deepcopy(landmark_list)
#     base_x, base_y = 0, 0
#     for i, pt in enumerate(tmp):
#         if i == 0:
#             base_x, base_y = pt[0], pt[1]
#         tmp[i][0] -= base_x
#         tmp[i][1] -= base_y
#     tmp = list(itertools.chain.from_iterable(tmp))
#     max_value = max(list(map(abs, tmp))) if tmp else 1.0
#     if max_value == 0:
#         return [0.0 for _ in tmp]
#     return [v / max_value for v in tmp]


# def pre_process_point_history(image, point_history):
#     image_width, image_height = image.shape[1], image.shape[0]
#     tmp = copy.deepcopy(point_history)
#     base_x, base_y = 0, 0
#     for i, p in enumerate(tmp):
#         if i == 0:
#             base_x, base_y = p[0], p[1]
#         tmp[i][0] = (tmp[i][0] - base_x) / image_width
#         tmp[i][1] = (tmp[i][1] - base_y) / image_height
#     return list(itertools.chain.from_iterable(tmp))


# def logging_csv(number, mode, landmark_list, point_history_list):
#     if mode == 0:
#         return
#     if mode == 1 and (0 <= number <= 9):
#         with open('model/keypoint_classifier/keypoint.csv', 'a', newline="") as f:
#             csv.writer(f).writerow([number, *landmark_list])
#     if mode == 2 and (0 <= number <= 9):
#         with open('model/point_history_classifier/point_history.csv', 'a', newline="") as f:
#             csv.writer(f).writerow([number, *point_history_list])


# def draw_landmarks(image, landmark_point):
#     if len(landmark_point) > 0:
#         # Thumb
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)
#         # Index finger
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)
#         # Middle finger
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)
#         # Ring finger
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)
#         # Little finger
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

#         # Palm
#         cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

#     for index, (x, y) in enumerate(landmark_point):
#         radius = 8 if index in (4, 8, 12, 16, 20) else 5
#         cv.circle(image, (x, y), radius, (255, 255, 255), -1)
#         cv.circle(image, (x, y), radius, (0, 0, 0), 1)
#     return image


# def draw_bounding_rect(use_brect, image, brect):
#     if use_brect:
#         cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
#                      (0, 0, 0), 1)
#     return image


# def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
#     cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
#                  (0, 0, 0), -1)
#     info_text = handedness.classification[0].label[0:]
#     if hand_sign_text != "":
#         info_text = info_text + ':' + hand_sign_text
#     cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
#                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
#     return image


# def draw_point_history(image, point_history):
#     for index, point in enumerate(point_history):
#         if point[0] != 0 and point[1] != 0:
#             cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
#                       (152, 251, 152), 2)
#     return image


# def draw_info(image, fps, mode, number):
#     cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
#                1.0, (0, 0, 0), 4, cv.LINE_AA)
#     cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
#                1.0, (255, 255, 255), 2, cv.LINE_AA)

#     mode_string = ['Logging Key Point', 'Logging Point History']
#     if 1 <= mode <= 2:
#         cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
#                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
#         if 0 <= number <= 9:
#             cv.putText(image, "NUM:" + str(number), (10, 110),
#                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
#     return image


# if __name__ == '__main__':
#     main()

