import json
import os
import random
import time

import paho.mqtt.client as mqtt

BROKER = os.getenv("MQTT_BROKER", "mqtt-broker")
PORT = int(os.getenv("MQTT_PORT", "1883"))
CLIENT_ID = os.getenv("CLIENT_ID", "iot-client-1")
TOPIC = os.getenv("MQTT_TOPIC", f"iot/{CLIENT_ID}/telemetry")
INTERVAL = float(os.getenv("PUBLISH_INTERVAL", "5"))

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id=CLIENT_ID
)

client.connect(BROKER, PORT, keepalive=60)
client.loop_start()

sequence = 0

try:
    while True:
        payload = {
            "device_id": CLIENT_ID,
            "sequence": sequence,
            "temperature": round(random.uniform(20.0, 30.0), 2),
            "humidity": round(random.uniform(35.0, 70.0), 2),
            "timestamp": time.time(),
        }

        client.publish(
            TOPIC,
            json.dumps(payload),
            qos=0,
            retain=False
        )

        print(json.dumps(payload), flush=True)

        sequence += 1
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    pass

finally:
    client.loop_stop()
    client.disconnect()
