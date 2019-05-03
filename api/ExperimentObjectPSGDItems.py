import sys
import os
import socket

HOME = os.environ['HOME']
sys.path.insert(1, HOME + '/github/StreamingSVM')
from api import Constant
from api import ExperimentObjectPSGD


class ExperimentObjectsPSGDItems:

    def getlist(self):
        datasets = ['webspam', 'cod-rna', 'phishing', 'w8a', 'a9a', 'ijcnn1']
        n_features = [Constant.Constant.WEBSPAM_F, Constant.Constant.COD_RNA_F, Constant.Constant.PHISHING_F,
                      Constant.Constant.W8A_F, Constant.Constant.A9A_F, Constant.Constant.IJCNN1_F]
        splits = [Constant.Constant.SPLIT_WEBSPAM, Constant.Constant.SPLIT_COD_RNA, Constant.Constant.SPLIT_PHISHING,
                  Constant.Constant.SPLIT_W8A, Constant.Constant.SPLIT_A9A,
                  Constant.Constant.SPLIT_IJCNN1]
        data_sources = [Constant.Constant.SOURCE_WEBSPAM, Constant.Constant.SOURCE_COD_RNA,
                        Constant.Constant.SOURCE_PHISHING, Constant.Constant.SOURCE_W8A,
                        Constant.Constant.SOURCE_A9A,Constant.Constant.SOURCE_IJCNN1]
        allsamples = [Constant.Constant.WEBSPAM_S, Constant.Constant.COD_RNA_S, Constant.Constant.PHISHING_S,
                      Constant.Constant.W8A_S, Constant.Constant.A9A_S, Constant.Constant.IJCNN1_S]
        training_files = [Constant.Constant.TRAINING_FILE_WEBSPAM, Constant.Constant.TRAINING_FILE_COD_RNA,
                          Constant.Constant.TRAINING_FILE_PHISHING, Constant.Constant.TRAINING_FILE_W8A,
                          Constant.Constant.TRAINING_FILE_A9A, Constant.Constant.TRAINING_FILE_IJCNN1]
        testing_files = [Constant.Constant.TESTING_FILE_WEBSPAM, Constant.Constant.TESTING_FILE_COD_RNA,
                         Constant.Constant.TESTING_FILE_PHISHING, Constant.Constant.TESTING_FILE_W8A,
                         Constant.Constant.TESTING_FILE_A9A, Constant.Constant.TESTING_FILE_IJCNN1]
        all_training_samples = [Constant.Constant.TRAINING_SAMPLES_WEBSPAM, Constant.Constant.TRAINING_SAMPLES_COD_RNA,
                                Constant.Constant.TRAINING_SAMPLES_PHISHING,
                                Constant.Constant.TRAINING_SAMPLES_W8A, Constant.Constant.TRAINING_SAMPLES_A9A,
                                Constant.Constant.TRAINING_SAMPLES_IJCNN1]
        all_testing_samples = [Constant.Constant.TESTING_SAMPLES_WEBSPAM, Constant.Constant.TESTING_SAMPLES_COD_RNA,
                               Constant.Constant.TESTING_SAMPLES_PHISHING,
                               Constant.Constant.TESTING_SAMPLES_W8A, Constant.Constant.TESTING_SAMPLES_A9A,
                               Constant.Constant.TESTING_SAMPLES_IJCNN1]

        DATA_SET = "cod-rna"
        DATA_SOURCE = Constant.Constant.SOURCE_COD_RNA
        FEATURES = Constant.Constant.COD_RNA_F
        SAMPLES = Constant.Constant.COD_RNA_S
        SPLIT = Constant.Constant.SPLIT_COD_RNA
        TRAINING_FILE = Constant.Constant.TRAINING_FILE_COD_RNA
        TESTING_FILE = Constant.Constant.TESTING_FILE_COD_RNA
        TRAINING_SAMPLES = Constant.Constant.TRAINING_SAMPLES_COD_RNA
        TESTING_SAMPLES = Constant.Constant.TESTING_SAMPLES_COD_RNA

        REPITITIONS = 10
        exps = []
        for data_set, features, split, data_source, samples, training_file, testing_file, training_samples, testing_samples in zip(
                datasets, n_features, splits, data_sources, allsamples, training_files, testing_files,
                all_training_samples,
                all_testing_samples):
            exp = ExperimentObjectPSGD.ExperimentObjectPSGD(dataset=data_set, data_source=data_source,
                                                            features=features,
                                                            samples=samples, split=split, training_file=training_file,
                                                            testing_file=testing_file,
                                                            training_samples=training_samples,
                                                            testing_samples=testing_samples, repititions=REPITITIONS)
            exps.append(exp)

        return exps