import pandas
import numpy
import re
import seaborn as sb

from matplotlib import pyplot as plt


class Visualizer:

    def __init__(self) -> None:
        self.main_frame = pandas.DataFrame()
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
        self.markers = ['d', 'X', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p']
        self.marker_size = 10
        self.width, self.height = plt.figaspect(1.4)
        self.fig = plt.figure(figsize=(self.width, self.height), dpi=1000, tight_layout=False)

    def set_data_frame(self):
        self.main_frame = pandas.read_csv('gender.csv')
        print('Break')

    def check_nan_columns(self):
        print("DataFrame Size before removing columns")
        print(self.main_frame.shape)

        # Remove Columns with all NaN values
        nan_columns = self.main_frame.columns[self.main_frame.isna().any()].tolist()
        for col in nan_columns:
            col = int(re.findall(r'\d$', col)[0])
            sum_na = self.main_frame.iloc[:, col].isna().sum()
            if sum_na == self.main_frame.shape[0]:
                self.main_frame.drop(self.main_frame.columns[col], axis=1, inplace=True)

        print("DataFrame Size after removing columns")
        print(self.main_frame.shape)

    def check_nan(self):
        bool_nan = self.main_frame.isnull().values.any()
        if bool_nan:
            print("There are NaN/NULL values in the dataset, removing them...")
            self.check_nan_columns()
            print("NaN values removed")
        else:
            print("There are no NaN/NULL values in the dataset")

    def strip_spaces(self):
        # Gather the Columns that have Strings as data type
        string_objects = self.main_frame.select_dtypes(['object'])

        # Strip trailing/leading whitespaces from the Strings
        self.main_frame[string_objects.columns] = string_objects.apply(lambda x: x.str.strip())
        print('String values stripped of trailing and leading whitespaces')

    def age_comparison_plot(self):
        self.main_frame.boxplot(column=' Age', by=' Gender', color='blue',
                                boxprops=dict(linewidth=3.0, color='blue'),
                                whiskerprops=dict(linestyle='-', linewidth=3.0, color='red'))
        plt.xlabel('Gender')
        plt.ylabel('Age')
        plt.suptitle('Age Comparisons')
        plt.savefig('./Report Images/Age Comparisons.png')

    def height_comparison_plot(self):
        self.main_frame.boxplot(column=' Height (cm)', by=' Gender', color='blue',
                                boxprops=dict(linewidth=3.0, color='blue'),
                                whiskerprops=dict(linestyle='-', linewidth=3.0, color='red'))
        plt.xlabel('Gender')
        plt.ylabel('Height')
        plt.suptitle('Height Comparisons')
        plt.savefig('./Report Images/Height Comparisons.png')

    def weight_comparison_plot(self):
        self.main_frame.boxplot(column=' Weight (kg)', by=' Gender', color='blue',
                                boxprops=dict(linewidth=3.0, color='blue'),
                                whiskerprops=dict(linestyle='-', linewidth=3.0, color='red'))
        plt.xlabel('Gender')
        plt.ylabel('Weight')
        plt.suptitle('Weight Comparisons')
        plt.savefig('./Report Images/Weight Comparisons.png')

    def income_comparison_plot(self):
        self.main_frame.boxplot(column=' Income (USD)', by=' Gender', color='blue',
                                boxprops=dict(linewidth=3.0, color='blue'),
                                whiskerprops=dict(linestyle='-', linewidth=3.0, color='red'))
        plt.xlabel('Gender')
        plt.ylabel('Income')
        plt.suptitle('Income Comparisons')
        plt.savefig('./Report Images/Income Comparisons.png')

    def corr_heatmap(self):
        sb.heatmap(self.main_frame.corr(), cmap="YlGnBu", annot=True)
        plt.savefig('./Report Images/Correlation Heatmap.png')

    def occupational_divide(self):
        male_occupation_numbers = self.main_frame[self.main_frame[' Gender'] == 'male'].groupby(' Occupation').size()
        female_occupation_numbers = self.main_frame[self.main_frame[' Gender'] == 'female'].groupby(
            ' Occupation').size()

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

        plt.savefig('./Report Images/Occupational Divide.png')
        print('Break')

    def educational_divide(self):
        male_occupation_numbers = self.main_frame[self.main_frame[' Gender'] == 'male'].groupby(
            ' Education Level').size()
        female_occupation_numbers = self.main_frame[self.main_frame[' Gender'] == 'female'].groupby(
            ' Education Level').size()

        shape_matched_counts = pandas.concat([male_occupation_numbers, female_occupation_numbers], axis=1)
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
        # plt.setp(axes.get_xticklabels(), rotation=30, horizontalalignment='right')
        axes.set_ylabel('Number of Males/Females')
        axes.set_title('Educational Divide amongst Men and Women')
        axes.legend()

        plt.savefig('./Report Images/Educational Divide.png')
        print('Break')
