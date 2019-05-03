ijcnn2csv:
	python operations/Libsvm2CSV.py /home/vibhatha/data/svm/ijcnn1/ijcnn1.tr /home/vibhatha/data/svm/ijcnn1/ijcnn1_training 22;python operations/Libsvm2CSV.py /home/vibhatha/data/svm/ijcnn1/ijcnn1.t /home/vibhatha/data/svm/ijcnn1/ijcnn1_testing 22

a9a2csv:
	python operations/Libsvm2CSV.py /home/vibhatha/data/svm/a9a/training.libsvm /home/vibhatha/data/svm/a9a/a9a_training 123;python operations/Libsvm2CSV.py /home/vibhatha/data/svm/a9a/testing.libsvm /home/vibhatha/data/svm/a9a/a9a_testing 123

covtype2csv:
	python operations/Libsvm2CSV.py /home/vibhatha/data/svm/covtype/training.libsvm /home/vibhatha/data/svm/covtype/training.csv 54
