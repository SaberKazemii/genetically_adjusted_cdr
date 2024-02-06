
import pandas as pd
import matplotlib.pyplot as plt

class DataAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.cleaned_metadata = None
        self.percentile_95_icd_case_0 = None
        self.percentile_95_icd_case_1 = None
        self.misclassified_as_1 = None
        self.misclassified_as_0 = None
        self.percent_icd_case_0 = None
        self.percent_icd_case_1 = None

    def load_data(self):
        self.cleaned_metadata = pd.read_csv(self.file_path)
        self.cleaned_metadata['redefined_icd_case'] = self.cleaned_metadata['worseeye_cdr'].apply(lambda x: 0 if x < 0.6 else 1)

    def calculate_statistics(self):
        self.percentile_95_icd_case_0 = self.cleaned_metadata[self.cleaned_metadata['redefined_icd_case'] == 0]['worseeye_cdr'].quantile(0.95)
        self.percentile_95_icd_case_1 = self.cleaned_metadata[self.cleaned_metadata['redefined_icd_case'] == 1]['worseeye_cdr'].quantile(0.95)

        original_icd_case_0 = self.cleaned_metadata[self.cleaned_metadata['icd_case'] == 0]
        original_icd_case_1 = self.cleaned_metadata[self.cleaned_metadata['icd_case'] == 1]

        self.misclassified_as_1 = original_icd_case_0[original_icd_case_0['worseeye_cdr'] > 0.6].shape[0] / original_icd_case_0.shape[0] * 100
        self.misclassified_as_0 = original_icd_case_1[original_icd_case_1['worseeye_cdr'] < 0.6].shape[0] / original_icd_case_1.shape[0] * 100

        total_cases = self.cleaned_metadata.shape[0]
        self.percent_icd_case_0 = original_icd_case_0.shape[0] / total_cases * 100
        self.percent_icd_case_1 = original_icd_case_1.shape[0] / total_cases * 100

    def plot_data(self):
        plt.figure(figsize=(10, 6))
        plt.hist(self.cleaned_metadata[self.cleaned_metadata['redefined_icd_case'] == 0]['worseeye_cdr'], bins=30, color='skyblue', edgecolor='black', alpha=0.5, label='icd_case = 0 (CDR < 0.6)')
        plt.hist(self.cleaned_metadata[self.cleaned_metadata['redefined_icd_case'] == 1]['worseeye_cdr'], bins=30, color='salmon', edgecolor='black', alpha=0.5, label='icd_case = 1 (CDR > 0.6)')
        plt.title('Distribution of Worse Eye CDR Values with Annotations')
        plt.xlabel('Worse Eye CDR')
        plt.ylabel('Frequency')
        plt.axvline(x=self.percentile_95_icd_case_0, color='blue', linestyle='dashed', linewidth=1)
        plt.axvline(x=self.percentile_95_icd_case_1, color='red', linestyle='dashed', linewidth=1)
        plt.text(self.percentile_95_icd_case_0, plt.gca().get_ylim()[1]*0.9, f'95th percentile (0.5)', color='blue', ha='center')
        plt.text(self.percentile_95_icd_case_1, plt.gca().get_ylim()[1]*0.8, f'95th percentile (0.95)', color='red', ha='center')
        plt.text(0.6, plt.gca().get_ylim()[1]*0.7, f'{self.misclassified_as_1:.2f}% of icd_case = 0 misclassified as 1', color='blue', ha='center')
        plt.text(0.6, plt.gca().get_ylim()[1]*0.6, f'{self.misclassified_as_0:.2f}% of icd_case = 1 misclassified as 0', color='red', ha='center')
        plt.text(0.6, plt.gca().get_ylim()[1]*0.5, f'{self.percent_icd_case_0:.2f}% are icd_case = 0', color='black', ha='center')
        plt.text(0.6, plt.gca().get_ylim()[1]*0.4, f'{self.percent_icd_case_1:.2f}% are icd_case = 1', color='black', ha='center')
        plt.grid(True)
        plt.legend()
        plt.show()
