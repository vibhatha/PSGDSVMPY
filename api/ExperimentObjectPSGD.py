

class ExperimentObjectPSGD:

    def __init__(self, dataset, data_source, features, samples, split, training_file, testing_file, training_samples, testing_samples, repititions, machine='ubuntu-monster'):
        self.dataset = dataset
        self.data_soruce = data_source
        self.features = features
        self.samples = samples
        self.split = split
        self.training_file = training_file
        self.testing_file =testing_file
        self.training_samples = training_samples
        self.testing_samples = testing_samples
        self.repititions = repititions
        self.machine = machine




