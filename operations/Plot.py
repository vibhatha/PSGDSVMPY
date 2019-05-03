import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np


class Plot:

    def __init__(self, logfile):
        self.logfile = logfile

    def sample(self):
        # Load the data
        fig, ax = plt.subplots()

        # Load in data
        tips = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")

        # Create violinplot
        ax.violinplot(tips["total_bill"], vert=False)

        # Show the plot
        plt.show()

    def load_sgd_data(self):
        df = pd.read_csv(self.logfile)
        alpha_df = df['alpha']
        epoch_df = df['epochs']
        acc_df = df['accuracy']
        time_df = df['time']
        print(alpha_df)
        max_acc = acc_df.max()
        min_time = time_df.min()
        max_acc_id = acc_df.idxmax()
        min_time_id = time_df.idxmin()
        print(max_acc, max_acc_id, min_time, min_time_id)
        return acc_df, time_df, max_acc, min_time, max_acc_id, min_time_id

    def single_plot(self, x, y):
        data = pd.DataFrame(data={'x': np.arange(0,len(x),1), 'y': y})

        # Create lmplot
        lm = sns.lmplot('x', 'y', data)

        # Get hold of the `Axes` objects
        axes = lm.ax

        # Tweak the `Axes` properties
        axes.set_ylim(0, 100 )

        # Show the plot
        plt.show()

dataset = "ijcnn1"
logFile = "logs/"+dataset+"_sgd_results.txt"
plot = Plot(logfile=logFile)
#plot.sample()
acc_df, time_df, max_acc, min_time, max_acc_id, min_time_id = plot.load_sgd_data()
plot.single_plot(time_df, acc_df)
