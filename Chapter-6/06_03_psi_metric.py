# Load new dataset of user data, which you've asyncronously updated 
import requests 


with requests.get('NEW DATASET') as r: 
    with open('new_dataset.csv', 'w+') as f: 
        f.write(r.text) 

import pandas as pd 
dataset = pd.read_csv("./new_dataset.csv") 

# Load historic dataset 
reference_dataset = pd.read_csv("./old_dataset")

# Define Report class, which is were we will run our evaluations
class Report: 
    def __init__(self, baseline, metrics=[]): 
        self.super 
        self.metrics = metrics 
        self.statistics = {} 
        self.baseline = baseline 

    def run(self, dataset): 
        for metric in self.metrics: 
            self.statistics[metric.name] = metric.run(self.baseline, dataset) 
        print("Report is ready!") 


class Metric: 
    def __init__(self, name, *args, **kwargs): 
        self.name = name 
        self.metric_kwargs = kwargs 

    def run(self, baseline, data): 
        pass 

class PSI(Metric): 
    def __init__(self, bucket_types='bins', buckets=10, axis=0): 
        super().__init__() 
        self.name = 'PSI' 
        self.bucket_types = bucket_types 
        self.buckets = buckets 
        self.axis = axis 

    def _calculate_psi(self, expected, actual): 
        """ 
        Calculate the PSI (population stability index) across all variables 
        """ 
        bucket_type = self.bucket_type 
        buckets = self.buckets 
        axis = self.axis 

        def psi(expected_array, actual_array, buckets): 
            def scale_range(inp, min, max): 
                inp += -(np.min(inp)) 
                inp /= np.max(inp) / (max - min) 
                inp += min 
                return inp 

            breakpoints = np.arange(0, buckets + 1) / (buckets) * 100 
            if bucket_type == 'bins': 
                breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array)) 
            elif bucket_type == 'quantiles': 
                breakpoints = np.percentile(expected_array, breakpoints) 
             
            expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array) 
            actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array) 

            def sub_psi(e_perc, a_perc): 
                """Calculate the actual PSI value from comparing the expected and actual distributions.""" 
                if a_perc == 0: 
                    return 0 
                elif e_perc == 0: 
                    return a_perc * np.log(a_perc / 0.01) 
                else: 
                    return (e_perc - a_perc) * np.log(e_perc / a_perc) 

            psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))) 
            return psi_value 

        psi_values = pd.Series(index=expected.columns) 
        for col in expected.columns: 
            psi_values[col] = psi(expected[col], actual[col], buckets=buckets, bucket_type=bucket_type)
        return psi_values 
 
    def run(self, baseline, data): 
        output = self._calculate_psi(baseline, data) 
        return output 