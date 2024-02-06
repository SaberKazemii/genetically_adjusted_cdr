
import pandas as pd
import matplotlib.pyplot as plt

class EyeConditionAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.cleaned_metadata = None

    def load_data(self):
        self.cleaned_metadata = pd.read_csv(self.file_path)

    def plot_frequency(self):
        frequency = self.cleaned_metadata['icd_case'].value_counts()

        bars = plt.bar(frequency.index, frequency.values, color=['skyblue', 'salmon'])
        plt.xlabel('Normal Eye (0) vs. Glaucomatous Eye (1)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Normal vs. Glaucomatous Eyes')
        plt.xticks([0, 1])

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')

        plt.show()
