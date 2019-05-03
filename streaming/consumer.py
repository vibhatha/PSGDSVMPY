from kafka import KafkaConsumer
#connect to Kafka server and pass the topic we want to consume
consumer = KafkaConsumer('test-counter-stream', group_id='view', bootstrap_servers=['0.0.0.0:9092'])


def consume():
    # return a multipart response
    data = kafkastream()


def kafkastream():

    for msg in consumer:
        #yield
        print(msg.value + '\n')

if __name__ == '__main__':
    consume()
