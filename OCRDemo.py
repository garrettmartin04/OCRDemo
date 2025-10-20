"""
Live OCR demo with smoothing:
 - captures webcam feed
 - preprocesses frames
 - runs Tesseract OCR
 - uses a rolling history (last 5 OCR results) 
 -  shows text that appears consistently (min 3 out of 5 frames)
"""

import cv2
import pytesseract
import numpy as np
from collections import deque, Counter, defaultdict
import json
import time
import requests

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR/tesseract.exe"  # path to tesseract executable

# Tesseract config 
TESSERACT_CONFIG = r'--oem 3 --psm 6'  # oem 3 = default engine(LSTM), psm 6 = block of text

# history of OCR results
history = deque(maxlen=5)



# predefined vocabulary (placeholder)
CARD_VOCAB = {"WIZARD", "DUCK", "HAND", "FIST", "HEALTH", "POINTS"}
# actions for each card (placeholder)
CARD_TO_ACTION = {
    "WIZARD": "cast_spell",
    "DUCK": "spawn_duck",
    "HAND": "show_hand",
    "FIST": "punch",
    "HEALTH": "heal_player",
    "POINTS": "add_points",
}


# normalize text for matching
def _norm(s: str) -> str:
    return (s or "").strip().upper()

# cleans up each frame for better OCR
def preprocess_for_ocr(frame):
    """
    preprocess image: grayscale, denoise, threshold, morphology
    I learned that Tesseract works better on grayscale images, removes color noise
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray image
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75) # denoise but keep edges
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu's thresholding: was cool to read about
    # morphological closing to close gaps in letters, only really matters for letters with tiny breaks
    kernel = np.ones((2, 2), np.uint8) 
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    return th


def run_ocr(ocr_image):
    """
    runs OCR on a preprocessed frame, returns list of words with confidence and bounding box
    """
    # check if format is correct for pytesseract (RGB)
    if len(ocr_image.shape) == 2:
        ocr_for_tess = cv2.cvtColor(ocr_image, cv2.COLOR_GRAY2RGB)
    else:
        ocr_for_tess = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2RGB)
    # data about each detected word
    data = pytesseract.image_to_data(
        ocr_for_tess,
        output_type=pytesseract.Output.DICT,
        config=TESSERACT_CONFIG
    )

    results = []
    # parse results, filter out empty/weak/confused results
    for i, text in enumerate(data['text']):
        text = text.strip()
        if text == "":
            continue
        try:
            conf = float(data['conf'][i])
        except:
            conf = -1 # sometimes conf is missing
        if conf < 20:  # too weak 
            continue
        # bounding box
        x, y, w, h = (
            data['left'][i],
            data['top'][i],
            data['width'][i],
            data['height'][i]
        )

        if _norm(text) not in CARD_VOCAB:
            continue

        results.append((_norm(text), conf, (x, y, w, h)))

    return results


def draw_smooth(frame, history, min_support = 3):
    """
    draw boxes for text that appears in >= 3 frames
    """
    # counts frequency of each recognized word across history
    all_results = [text for frame_res in history for (text, conf, box) in frame_res]
    counts = Counter(all_results) # ++

    # boxes for the latest frame results, if stable
    for frame_res in history[-1]: 
        text, conf, (x, y, w, h) = frame_res
        if counts[text] >= min_support:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{text} ({int(conf)})" # show confidence too
            # label above box, not offscreen
            cv2.putText( 
                frame,
                label,
                (x, max(y - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            # returns display image
    return frame
    # JSON events
def stable_events(history, frame_idx, min_support=3):
    if not history: # needed
        return []
    #counter and confidence lists
    counts = Counter()
    confs = defaultdict(list)
    last_bbox = {}
    # iterate through history
    for frame_res in history:
        seen = set() # avoid double counting in same frame
        for t, c, b in frame_res:
            if t in seen:
                continue
            seen.add(t)
            counts[t] += 1
            confs[t].append(c)
            last_bbox[t] = b
    events = [] # list of events
    for t, n in counts.items(): # checking if stable
        if n >= min_support:
            x, y, w, h = last_bbox.get(t, (0, 0, 0, 0))
            avg_conf = float(np.mean(confs[t])) if confs[t] else 0.0
            events.append({ # structure of event
                "eventType": "card",
                "value": t,
                "confidence": round(avg_conf, 2),
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "countInWindow": int(n)
            })
    return events


API_BASE = "http://127.0.0.1:8000"  # server
session = requests.Session()         # reusing TCP connection (faster)

def post_event(event: dict) -> bool:
    """
    Send a single OCR event to POST /event.
    Returns True if the server responded OK, else False (non-fatal).
    """
    try:
        payload = dict(event)          
        payload["timestamp"] = time.time()  # order
        resp = session.post(f"{API_BASE}/event", json=payload, timeout=0.5)
        resp.raise_for_status()       # errors
        return True
    except requests.RequestException as e:
        # crash log
        print(f"[WARN] POST /event failed: {e}")
        return False



def to_game_event(evt: dict) -> dict:
    """
    converts a raw OCR 'card' event to a game event
    keeps backward compatibility (original keys) and adds action and metadata
    """
    token = (evt.get("value") or "").strip().upper()
    action = CARD_TO_ACTION.get(token)  

    return {
        # OG fields
        "eventType": evt.get("eventType", "card"),
        "value": token,
        "confidence": evt.get("confidence"),
        "bbox": evt.get("bbox"),
        "countInWindow": evt.get("countInWindow"),

        # new 
        "action": action,
        "source": "ocr-demo",
        "schemaVersion": 1,
        "timestamp": time.time(),         # client time
    }



def main():
    seen_words = set()  # track words already printed
    cap = cv2.VideoCapture(1)  # webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # set height
    # needed
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    frame_count = 0 # count frames
    PROCESS_EVERY_N = 8  # OCR every N frames
    emitted_tokens = set()

    while True:
        ret, frame = cap.read() # read frame
        # needed
        if not ret:
            break

        display_frame = frame.copy()  # copy frame so we can draw on it, preserving original

        if frame_count % PROCESS_EVERY_N == 0:
            # on every 5th frame we:
            ocr_img = preprocess_for_ocr(frame) # preprocess
            results = run_ocr(ocr_img) # OCR
            history.append(results) # add to history

            events = stable_events(history, frame_count, min_support=3)
            for evt in events:
                t = evt["value"]
                if t not in emitted_tokens:
                    data = to_game_event(evt) # more
                    print(json.dumps(data))
                    post_event(data) # post events to backbone
                    emitted_tokens.add(t)

        # draw results from history
        if history:
            display_frame = draw_smooth(display_frame, history, min_support = 3)

        cv2.imshow("Live OCR (demo)", display_frame) # show frame on screen

        key = cv2.waitKey(1) & 0xFF # required for imshow to work
        if key == ord('q'):  # end on q
            break

        frame_count += 1 #iterate
    # clean
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()