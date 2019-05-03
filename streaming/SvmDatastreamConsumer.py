import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from sgd import SVM
import numpy as np
from kafka import KafkaConsumer


class SvmDatastreamConsumer:

    def __init__(self, topics=[]):
        self.topics = topics


    def consume(self):
        for topic in self.topics:
            consumer = KafkaConsumer(topic, group_id='view', bootstrap_servers=['0.0.0.0:9092'])
            chunk_id = 0
            for msg in consumer:
                # yield
                dt = np.dtype(np.float64)
                arr = np.frombuffer(msg.value, dtype=dt)
                print("")
                print("###### Reading Chunk " + str(chunk_id) + "########")
                line = ""
                for i in range(0, len(arr)):
                    line += str(arr[i]) + " "
                print(line)
                print("#################################################")
                chunk_id = chunk_id + 1

topic_x_training = 'svm_training_data_stream_x'
topic_y_training = 'svm_training_data_stream_y'
topic_x_testing = 'svm_testing_data_stream_x'
topic_y_testing = 'svm_testing_data_stream_y'
topics = [topic_x_training, topic_y_training, topic_x_testing, topic_x_testing]
streamConsumer = SvmDatastreamConsumer(topics=topics)
streamConsumer.consume()
