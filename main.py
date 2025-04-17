import numpy as np

rf_model = None

data = np.random.randint(0, 100, 100)

abundance_cutoff = 0.00001
prevalence_cutoff = 0.05

mean_abundance = None  # TODO

def clr_transform(data, log_n0: float = 1e-06):
    pass

def predict(model, data, type):
    return data


