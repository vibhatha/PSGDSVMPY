# producer.py

import time
import cv2
from kafka import SimpleProducer
from kafka import SimpleClient
#  connect to Kafka
kafka = SimpleClient('localhost:9092')
producer = SimpleProducer(kafka)
# Assign a topic
topic = 'test-counter-stream'


def stream_emitter(count_max):    # Open the video

    print(' Streaming.....')
    count = 0
    # read the file
    while (count < count_max):
        # read the image in each frame
        count_bytes = str.encode(str(count))
        producer.send_messages(topic, count_bytes)
        # To reduce CPU usage create sleep time of 0.2sec
        count = count + 1
        time.sleep(2)

    print('Done streaming')


if __name__ == '__main__':
    stream_emitter(100)
