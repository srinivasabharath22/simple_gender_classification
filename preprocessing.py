import pandas
import re
from sklearn.preprocessing import OneHotEncoder


class PreProcessing:
    def __init__(self):
        self.main_frame = pandas.DataFrame()
        self.x_block = None
        self.y_block = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def set_data_frame(self):
        # Set Main Frame after reading csv
        self.main_frame = pandas.read_csv('gender.csv')

    def check_nan_columns(self):
        print("DataFrame Size before removing columns")
        print(self.main_frame.shape)

        # Get columns with NaN values in the form of list
        nan_columns = self.main_frame.columns[self.main_frame.isna().any()].tolist()

        # Iterate through all the columns
        for col in nan_columns:
            col = int(re.findall(r'\d$', col)[0])
            sum_na = self.main_frame.iloc[:, col].isna().sum()
            # Drop a column if all the values ar NaN
            if sum_na == self.main_frame.shape[0]:
                self.main_frame.drop(self.main_frame.columns[col], axis=1, inplace=True)

        print("DataFrame Size after removing columns")
        print(self.main_frame.shape)

    def check_nan(self):
        # Check if NaN values are there in the main data frame
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

    def feature_label_splitter(self):
        # Drop the label column from features frame
        self.x_block = self.main_frame.copy()
        self.x_block.drop([' Gender'], axis=1, inplace=True)

        # Drop all the other columns from Label frame
        self.y_block = self.main_frame.copy()
        self.y_block = self.y_block[[' Gender']]

    def encoding_categorical(self):
        # Initialise Encoder
        onehotencoder = OneHotEncoder()

        # Get all the Categorical Columns
        string_objects = self.x_block.select_dtypes(['object'])

        # Encode those columns
        encoded_columns = onehotencoder.fit_transform(self.x_block[string_objects.columns]).toarray()

        # Create a dataframe with encoded columns
        onehot_df = pandas.DataFrame(encoded_columns, columns=onehotencoder.get_feature_names_out(string_objects.columns))

        # Drop the still categorical columns from the original dataframe
        self.x_block.drop(string_objects.columns, axis=1, inplace=True)

        # Concatenate the Encoded Columns in the original frame
        self.x_block = pandas.concat([self.x_block, onehot_df], axis=1)
