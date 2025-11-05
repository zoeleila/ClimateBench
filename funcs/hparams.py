from ClimateBench.funcs.settings import RAW_DIR, RUNS_DIR, DATASET_DIR

class HParams():
    def __init__(self):
        self.batch_size = 16
        self.epoch = 30
        self.learning_rate = 0.001
        self.slider = 10
        self.data_path = RAW_DIR / 'train_val'
        self.sample_dir = DATASET_DIR / 'dataset_exp1'
        self.runs_dir = RUNS_DIR
        self.model = 'lstm'
        self.var_to_predict = 'tas'
        self.exp = f'exp1/lstm_5scenarios_{self.var_to_predict}'
        self.vars = ['CO2', 'CH4', 'SO2', 'BC']
        self.len_historical = 165
        self.simus_train = ['ssp126',
                    'ssp370',
                    'ssp585',
                    'hist-GHG',
                    'hist-aer']
        self.simus_test = ['ssp245']
