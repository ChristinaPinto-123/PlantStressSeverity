"""Gateway (Raspberry Pi class): receives edge messages over MQTT, runs the heavier model on
uncertain scans, buffers offline, smooths with EWMA and applies the decision layer.

Topics (node id = plant id):
    leaf/<id>/result     {"cls","p","batt","q","temp","hum"}   confident edge result
    leaf/<id>/uncertain  {"spec":[18 floats],"temp","hum"}      low-confidence -> second opinion
    leaf/<id>/decision   published by this gateway
"""
import json
import sqlite3
import time

import joblib
import numpy as np

from config import CLASSES
from gate import EWMASmoother, decision_map

try:
    import paho.mqtt.client as mqtt
except ImportError:  # allow import for testing without paho
    mqtt = None

MODEL_PATH = "gateway_model.joblib"        # trained RF pipeline (joblib.dump(build_gateway(...).fit(...)))
BROKER = "localhost"

db = sqlite3.connect("buffer.db", check_same_thread=False)   # offline buffer + relabel queue
db.execute("CREATE TABLE IF NOT EXISTS scans(ts REAL, plant TEXT, tier INT, cls INT, conf REAL, "
           "temp REAL, hum REAL, spec TEXT, synced INT DEFAULT 0)")
smoother = EWMASmoother()
gateway_model = None


def handle(client, userdata, msg):
    global gateway_model
    _, plant, kind = msg.topic.split("/")
    d = json.loads(msg.payload)
    if kind == "uncertain":
        if gateway_model is None:
            gateway_model = joblib.load(MODEL_PATH)
        x = np.r_[d["spec"], d["temp"], d["hum"]].reshape(1, -1)
        p = gateway_model.predict_proba(x)[0]
        tier, spec = 1, json.dumps(d["spec"])      # raw spectrum kept for cloud relabeling
    else:
        p = np.full(len(CLASSES), (1 - d["p"]) / (len(CLASSES) - 1)); p[d["cls"]] = d["p"]
        tier, spec = 0, None
    db.execute("INSERT INTO scans VALUES(?,?,?,?,?,?,?,?,0)",
               (time.time(), plant, tier, int(p.argmax()), float(p.max()), d["temp"], d["hum"], spec))
    db.commit()
    sm = smoother.update(plant, p)
    sev = int(sm.argmax())
    action = decision_map(sev, d["temp"], d["hum"])
    client.publish(f"leaf/{plant}/decision",
                   json.dumps({"severity": CLASSES[sev], "p": float(sm.max()), "action": action}))


def sync_to_cloud(upload_fn):
    """Push unsynced rows when connectivity returns (upload_fn: your cloud API call)."""
    rows = db.execute("SELECT rowid,* FROM scans WHERE synced=0").fetchall()
    if rows and upload_fn(rows):
        db.execute("UPDATE scans SET synced=1 WHERE synced=0"); db.commit()


if __name__ == "__main__":
    c = mqtt.Client()
    c.on_message = handle
    c.connect(BROKER, 1883)
    c.subscribe("leaf/+/result"); c.subscribe("leaf/+/uncertain")
    c.loop_forever()
