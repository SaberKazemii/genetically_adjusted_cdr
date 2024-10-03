
import pandas as pd

class DemographicAnalysis:
    def __init__(self, file_path):
        self.cleaned_metadata = pd.read_csv(file_path)
        self.female_ratio = None
        self.mean_age = None
        self.std_age = None
        self.race_proportions = None

    def process_data(self):
        self.cleaned_metadata['sex_numeric'] = self.cleaned_metadata['sex'].apply(lambda x: 1 if x == 'Male' else 0)
        self.female_ratio = 1 - (sum(self.cleaned_metadata['sex_numeric']) / self.cleaned_metadata.shape[0])
        self.mean_age = self.cleaned_metadata['age'].mean()
        self.std_age = self.cleaned_metadata['age'].std()
        race_counts = self.cleaned_metadata['ancestry'].value_counts()
        self.race_proportions = race_counts / self.cleaned_metadata.shape[0]

    def print_statistics(self):
        print(f"The ratio of female sex is: {self.female_ratio:.2f}")
        print(f"The mean of age is: {self.mean_age:.2f} and the standard deviation is: {self.std_age:.2f}")
        print("The proportion of each race in the dataframe:")
        for race, proportion in self.race_proportions.items():
            print(f"{race}: {proportion:.2f}")
