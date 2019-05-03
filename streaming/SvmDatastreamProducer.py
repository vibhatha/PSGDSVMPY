import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from operations import LoadLibsvm
import time
from kafka import SimpleProducer
from kafka import SimpleClient
import numpy as np


class SvmDatastreamProducer:

    def __init__(self, training_file, testing_file):
        self.training_file = training_file
        self.testing_file = testing_file

    def load_svm_data(self):
        training_filepath = self.training_file
        testing_filepath = self.testing_file
        training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=22)
        testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_filepath, n_features=22)
        x_training, y_training = training_loader.load_all_data()
        x_testing, y_testing = testing_loader.load_all_data()
        return x_training, y_training, x_testing, y_testing

    def stream_training_data(self):
        #  connect to Kafka
        kafka = SimpleClient('localhost:9092')
        producer = SimpleProducer(kafka)
        # Assign a topic
        topic_x_training = 'svm_training_data_stream_x'
        topic_y_training = 'svm_training_data_stream_y'
        topic_x_testing = 'svm_testing_data_stream_x'
        topic_y_testing = 'svm_testing_data_stream_y'

        x_training, y_training, x_testing, y_testing = self.load_svm_data()
        print(x_training.shape, x_testing.shape)
        training_chunk_size = x_training.shape[0]
        testing_chunk_size = x_testing.shape[0]
        x_training_chunks = np.array_split(x_training, training_chunk_size)
        x_testing_chunks = np.array_split(x_testing, testing_chunk_size)
        y_training_chunks = np.array_split(y_training, training_chunk_size)
        y_testing_chunks = np.array_split(y_testing, testing_chunk_size)

        self.streamer(chunk_size=training_chunk_size, x=x_training_chunks, y=y_training_chunks,
                      x_topic=topic_x_training, y_topic=topic_y_training, time_lap=1, producer=producer)



        print("Streaming Completed.")

    def streamer(self, chunk_size, x, y, x_topic, y_topic, time_lap=0.1, producer=None):
        chunk_count = 0
        # stream training data
        while (chunk_count < chunk_size):
            print("Streaming Chunk : " + str(chunk_count + 1) + "/" + str(chunk_size))
            x_training_chunk = x[chunk_count]
            y_training_chunk = y[chunk_count]
            # print(x_training_chunk.dtype)
            # print(len(x_training_chunk), len(y_training_chunk), len(x_testing_chunk), len(y_testing_chunk))
            x_tr_bytes = x_training_chunk.tobytes()
            y_tr_bytes = y_training_chunk.tobytes()

            producer.send_messages(x_topic, x_tr_bytes)
            producer.send_messages(y_topic, y_tr_bytes)
            # To reduce CPU usage create sleep time of 0.2sec
            chunk_count = chunk_count + 1
            time.sleep(time_lap)


streamproducer = SvmDatastreamProducer(training_file="/home/vibhatha/data/svm/ijcnn1/ijcnn1_training",
                                       testing_file="/home/vibhatha/data/svm/ijcnn1/ijcnn1_testing")
streamproducer.stream_training_data()
