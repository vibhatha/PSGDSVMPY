from mpi4py import MPI

class Constant:

    JULIET_HOME = '/N/u/vlabeyko/data/svm/'
    UBUNTU_MONSTER_HOME = '/home/vibhatha/data/'
    HOME = UBUNTU_MONSTER_HOME

    MPI_INT = 'i'
    MPI_FLOAT = 'f'
    REAL_SLIM_S = 72309
    REAL_SLIM_F = 20958
    SOURCE_REAL_SIM = '/home/vibhatha/data/svm/real-slim/training.csv'
    WEBSPAM_S = 350000
    WEBSPAM_F = 254
    SOURCE_WEBSPAM = '/home/vibhatha/data/svm/webspam/training.csv'

    datasets = ['webspam', 'cod-rna', 'phishing', 'w8a', 'a9a', 'ijcnn1']
    n_features = [254, 8, 68, 300, 123, 22]
    splits = [True, False, True, False, False, False]

    COD_RNA_S = 59535
    COD_RNA_F = 8
    SOURCE_COD_RNA = HOME + 'svm/cod-rna/training.csv'
    TRAINING_FILE_COD_RNA = HOME +  'svm/cod-rna/training.csv'
    TESTING_FILE_COD_RNA = HOME +  'svm/cod-rna/training.csv'
    SPLIT_COD_RNA = False
    TRAINING_SAMPLES_COD_RNA = 59535
    TESTING_SAMPLES_COD_RNA = 271617

    WEBSPAM_S = 350000
    WEBSPAM_F = 254
    SOURCE_WEBSPAM = HOME + 'svm/webspam/training.csv'
    TRAINING_FILE_WEBSPAM = HOME + 'svm/webspam/training.csv'
    TESTING_FILE_WEBSPAM = None
    SPLIT_WEBSPAM = True
    TRAINING_SAMPLES_WEBSPAM = 280000
    TESTING_SAMPLES_WEBSPAM = 70000

    PHISHING_S = 11055
    PHISHING_F = 68
    SOURCE_PHISHING = HOME + 'svm/phishing/training.csv'
    TRAINING_FILE_PHISHING = HOME + 'svm/phishing/training.csv'
    TESTING_FILE_PHISHING = None
    SPLIT_PHISHING = True
    TRAINING_SAMPLES_PHISHING = 8844
    TESTING_SAMPLES_PHISHING = 2211

    W8A_S = 49749
    W8A_F = 300
    SOURCE_W8A = HOME + 'svm/w8a/training.csv'
    TRAINING_FILE_W8A = HOME + 'svm/w8a/training.csv'
    TESTING_FILE_W8A = HOME + 'svm/w8a/testing.csv'
    SPLIT_W8A = False
    TRAINING_SAMPLES_W8A = 49749
    TESTING_SAMPLES_W8A = 14951

    A9A_S = 32561
    A9A_F = 300
    SOURCE_A9A = HOME + 'svm/a9a/training.csv'
    TRAINING_FILE_A9A = HOME + 'svm/a9a/training.csv'
    TESTING_FILE_A9A = HOME + 'svm/a9a/testing.csv'
    SPLIT_A9A = False
    TRAINING_SAMPLES_A9A = 32561
    TESTING_SAMPLES_A9A = 16281

    IJCNN1_S = 32561
    IJCNN1_F = 300
    SOURCE_IJCNN1 = HOME + 'svm/ijcnn1/training.csv'
    TRAINING_FILE_IJCNN1 = HOME + 'svm/ijcnn1/training.csv'
    TESTING_FILE_IJCNN1 = HOME + 'svm/ijcnn1/testing.csv'
    SPLIT_IJCNN1 = False
    TRAINING_SAMPLES_IJCNN1 = 49990
    TESTING_SAMPLES_IJCNN1 = 91701







    def get_type(self, mpi_type=MPI.INT):
        if(mpi_type == MPI.INT):
            return self.MPI_INT
        if(mpi_type == MPI.FLOAT):
            return self.MPI_FLOAT
