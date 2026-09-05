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
ROLE = os.getenv("CLIENT_ROLE", "publisher").lower()
SUBSCRIBE_TOPIC = os.getenv("SUBSCRIBE_TOPIC", "iot/#")

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id=CLIENT_ID
)

def on_message(client, userdata, message):
    print(
        json.dumps({
            "topic": message.topic,
            "payload": message.payload.decode(
                "utf-8",
                errors="replace",
            ),
        }),
        flush=True,
    )


client.on_message = on_message
client.connect(BROKER, PORT, keepalive=60)

if ROLE == "subscriber":
    client.subscribe(SUBSCRIBE_TOPIC, qos=0)

client.loop_start()

sequence = 0

try:
    while True:
        if ROLE == "publisher":
            payload = {
                "device_id": CLIENT_ID,
                "sequence": sequence,
                "temperature": round(
                    random.uniform(20.0, 30.0),
                    2,
                ),
                "humidity": round(
                    random.uniform(35.0, 70.0),
                    2,
                ),
                "timestamp": time.time(),
            }

            client.publish(
                TOPIC,
                json.dumps(payload),
                qos=0,
                retain=False,
            )

            print(json.dumps(payload), flush=True)
            sequence += 1

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    pass

finally:
    client.loop_stop()
    client.disconnect()
