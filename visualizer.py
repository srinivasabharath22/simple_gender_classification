import pandas
import numpy
import seaborn as sb
from matplotlib import pyplot as plt
from preprocessing import PreProcessing


class Visualizer(PreProcessing):
    def __init__(self) -> None:
        super().__init__()

        # Initialising base plt parameters independent of plots
        plt.style.use('seaborn-whitegrid')
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.serif'] = 'Ubuntu'
        plt.rcParams['font.monospace'] = 'Ubuntu Mono'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 12
        self.width, self.height = plt.figaspect(1.4)
        self.fig = plt.figure(figsize=(self.width, self.height), dpi=1000, tight_layout=False)

    def age_comparison_plot(self):
        # Creating a Box_whisker plot for Age Comparison amongst classes with suitable changes to plot attributes
        self.main_frame.boxplot(column=' Age', by=' Gender', color='blue',
                                boxprops=dict(linewidth=3.0, color='blue'),
                                whiskerprops=dict(linestyle='-', linewidth=3.0, color='red'))
        plt.xlabel('Gender')
        plt.ylabel('Age')
        plt.suptitle('Age Comparisons')

        # Saving image for future use
        plt.savefig('./Report Images/Age Comparisons.png')

    def height_comparison_plot(self):
        # Creating a Box_whisker plot for Height Comparison amongst classes with suitable changes to plot attributes
        self.main_frame.boxplot(column=' Height (cm)', by=' Gender', color='blue',
                                boxprops=dict(linewidth=3.0, color='blue'),
                                whiskerprops=dict(linestyle='-', linewidth=3.0, color='red'))
        plt.xlabel('Gender')
        plt.ylabel('Height')
        plt.suptitle('Height Comparisons')

        # Saving image for future use
        plt.savefig('./Report Images/Height Comparisons.png')

    def weight_comparison_plot(self):
        # Creating a Box_whisker plot for Weight Comparison amongst classes with suitable changes to plot attributes
        self.main_frame.boxplot(column=' Weight (kg)', by=' Gender', color='blue',
                                boxprops=dict(linewidth=3.0, color='blue'),
                                whiskerprops=dict(linestyle='-', linewidth=3.0, color='red'))
        plt.xlabel('Gender')
        plt.ylabel('Weight')
        plt.suptitle('Weight Comparisons')

        # Saving image for future use
        plt.savefig('./Report Images/Weight Comparisons.png')

    def income_comparison_plot(self):
        # Creating a Box_whisker plot for Income Comparison amongst classes with suitable changes to plot attributes
        self.main_frame.boxplot(column=' Income (USD)', by=' Gender', color='blue',
                                boxprops=dict(linewidth=3.0, color='blue'),
                                whiskerprops=dict(linestyle='-', linewidth=3.0, color='red'))
        plt.xlabel('Gender')
        plt.ylabel('Income')
        plt.suptitle('Income Comparisons')

        # Saving image for future use
        plt.savefig('./Report Images/Income Comparisons.png')

    def corr_heatmap(self):
        # Create a Correlation heatmap for float values to see the importance of certain variables
        sb.heatmap(self.main_frame.corr(), cmap="YlGnBu", annot=True)
        plt.savefig('./Report Images/Correlation Heatmap.png')
        plt.title('HeatMap showing correlation amongst variables')

    def occupational_divide(self):
        # Get Occupation and corresponding numbers for both classes
        male_occupation_numbers = self.main_frame[self.main_frame[' Gender'] == 'male'].groupby(' Occupation').size()
        female_occupation_numbers = self.main_frame[self.main_frame[' Gender'] == 'female'].groupby(
            ' Occupation').size()

        # To match the shapes for plotting, pad the Categories where Male/Female classes have no data points with zeroes
        shape_matched_counts = pandas.concat([male_occupation_numbers, female_occupation_numbers], axis=1)
        shape_matched_counts.columns = ['male', 'female']
        shape_matched_counts.fillna(0, inplace=True)

        figure, axes = plt.subplots(dpi=1000, tight_layout=False)
        x_axis = numpy.arange(len(shape_matched_counts))
        width = 0.2
        axes.bar(x_axis, shape_matched_counts['male'], label='male', color='seagreen')
        axes.bar(x_axis, shape_matched_counts['female'], bottom=shape_matched_counts['male'], label='female',
                 color='darkred')
        axes.set_xlabel('Occupation')
        axes.set_xticks(x_axis + width / 2)
        axes.set_xticklabels(shape_matched_counts.index, fontsize=5)
        plt.setp(axes.get_xticklabels(), rotation=30, horizontalalignment='right')
        axes.set_ylabel('Number of Male/Female in Occupation')
        axes.set_title('Occupational Divide amongst Men and Women')
        axes.legend()

        # Saving image for future use
        plt.savefig('./Report Images/Occupational Divide.png')
        print('Break')

    def educational_divide(self):
        # Get Education and corresponding numbers for both classes
        male_educational_numbers = self.main_frame[self.main_frame[' Gender'] == 'male'].groupby(
            ' Education Level').size()
        female_educational_numbers = self.main_frame[self.main_frame[' Gender'] == 'female'].groupby(
            ' Education Level').size()

        # To match the shapes for plotting, pad the Categories where Male/Female classes have no data points with zeroes
        shape_matched_counts = pandas.concat([male_educational_numbers, female_educational_numbers], axis=1)
        shape_matched_counts.columns = ['male', 'female']
        shape_matched_counts.fillna(0, inplace=True)

        figure, axes = plt.subplots(dpi=1000, tight_layout=False)
        x_axis = numpy.arange(len(shape_matched_counts))
        width = 0.4
        axes.bar(x_axis, shape_matched_counts['male'], color='seagreen', width=width, label='male')
        axes.bar(x_axis + width, shape_matched_counts['female'], width=width, label='female', color='darkred')
        axes.set_xlabel('Education Level')
        axes.set_xticks(x_axis + width / 2)
        axes.set_xticklabels(shape_matched_counts.index, fontsize=8)
        axes.set_ylabel('Number of Males/Females')
        axes.set_title('Educational Divide amongst Men and Women')
        axes.legend()

        # Saving image for future use
        plt.savefig('./Report Images/Educational Divide.png')
        print('Break')

    def marital_divide(self):
        # Get Marital Status and corresponding numbers for both classes
        male_marital_numbers = self.main_frame[self.main_frame[' Gender'] == 'male'].groupby(
            ' Marital Status').size()
        female_marital_numbers = self.main_frame[self.main_frame[' Gender'] == 'female'].groupby(
            ' Marital Status').size()

        # To match the shapes for plotting, pad the Categories where Male/Female classes have no data points with zeroes
        shape_matched_counts = pandas.concat([male_marital_numbers, female_marital_numbers], axis=1)
        shape_matched_counts.columns = ['male', 'female']
        shape_matched_counts.fillna(0, inplace=True)

        figure, axes = plt.subplots(dpi=1000, tight_layout=False)
        x_axis = numpy.arange(len(shape_matched_counts))
        width = 0.4
        axes.bar(x_axis, shape_matched_counts['male'], color='seagreen', width=width, label='male')
        axes.bar(x_axis + width, shape_matched_counts['female'], width=width, label='female', color='darkred')
        axes.set_xlabel('Marital Status')
        axes.set_xticks(x_axis + width / 2)
        axes.set_xticklabels(shape_matched_counts.index, fontsize=8)
        axes.set_ylabel('Number of Males/Females')
        axes.set_title('Marital Status amongst Men and Women in the DataSet')
        axes.legend()

        # Saving image for future use
        plt.savefig('./Report Images/Marital Status Divide.png')
        print('Break')

    def favorite_color(self):
        # Get Favorite Color and corresponding numbers for both classes
        male_color_numbers = self.main_frame[self.main_frame[' Gender'] == 'male'].groupby(
            ' Favorite Color').size()
        female_color_numbers = self.main_frame[self.main_frame[' Gender'] == 'female'].groupby(
            ' Favorite Color').size()

        # To match the shapes for plotting, pad the Categories where Male/Female classes have no data points with zeroes
        shape_matched_counts = pandas.concat([male_color_numbers, female_color_numbers], axis=1)
        shape_matched_counts.columns = ['male', 'female']
        shape_matched_counts.fillna(0, inplace=True)

        figure, axes = plt.subplots(dpi=1000, tight_layout=False)
        x_axis = numpy.arange(len(shape_matched_counts))
        width = 0.4
        axes.bar(x_axis, shape_matched_counts['male'], color='seagreen', width=width, label='male')
        axes.bar(x_axis + width, shape_matched_counts['female'], width=width, label='female', color='darkred')
        axes.set_xlabel('Colors')
        axes.set_xticks(x_axis + width / 2)
        axes.set_xticklabels(shape_matched_counts.index, fontsize=8)
        axes.set_ylabel('Number of Males/Females')
        axes.set_title('Favorite Colors amongst Men and Women in the DataSet')
        axes.legend()

        # Saving image for future use
        plt.savefig('./Report Images/Favorite Colors.png')
        print('Break')
