class DecisionTree(object):

    def __init__(self):
        self.tree = None
        self.trained = False

    def fit(self, data_frame):
        if not isinstance(data_frame, pd.core.frame.DataFrame):
            raise ValueError('Please provide data_frame as an instance of pandas '+
                'DataFrame. Recieved data_frame as an instance of {}'.format(type(data_frame)))
        self.tree = self.buildTree(data_frame)
        self.trained = True
        print('Decision Tree trained successfully.')

    def buildTree(self, df, tree=None):
        """
        Here we build our decision tree
        """
        # Target variable
        Class = df.keys()[-1]
        # Get attribute with maximum information gain
        node = self.find_max_ig(df)
        # Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
        attValue = np.unique(df[node])
        # Create an empty dictionary to create tree
        if tree is None:                 
            tree={}
            tree[node] = {}
        # We make loop to construct a tree by calling this function recursively.
        # In this we check if the subset is pure and stops if it is pure.
        for value in attValue:
            subtable = self.get_subtable(df, node, value)
            clValue, counts = np.unique(subtable['Eat'], return_counts=True)
            # Checking purity of subset
            if len(counts)==1:
                # Subset is pure
                tree[node][value] = clValue[0]
            else:
                # Calling the function recursively
                tree[node][value] = self.buildTree(subtable)
        return tree

    def predict(self, X_test):
        if not self.trained:
            raise ValueError('You have to train the model before predicting. Use `model.fit(data)` to train')
        if not isinstance(X_test, pd.core.frame.DataFrame) \
            and not isinstance(X_test, pd.core.series.Series):
            raise ValueError('Please provide X_test as an instance of pandas '+
                'Series or DataFrame. Recieved X_test as an instance of {}'.format(type(X_test)))
        # If 1 sample is given
        if isinstance(X_test, pd.core.series.Series):
            return self.predict_sample(X_test, self.tree)
        # If multiple samples are given
        predictions = []
        for i in range(len(X_test)):
            predictions.append(self.predict_sample(X_test.iloc[i], self.tree))
        return predictions

    def predict_sample(self, sample, tree):
        """
        This function is used to predict for any input sample(s) 
        Recursively we go through the tree that we built during training
        """
        for nodes in tree.keys():  
            value = sample[nodes]
            tree = tree[nodes][value]
            prediction = 0    
            if type(tree) is dict:
                prediction = self.predict_sample(sample, tree)
            else:
                prediction = tree
                break
        return prediction

    def find_max_ig(self, df):
        """
        Given a dataframe, find maximum information gain among all attributes
        """
        ig_attributes = []
        attributes = df.keys()[:-1]
        for attribute in attributes:
            gain = self.find_entropy(df) - self.find_entropy_attribute(df, attribute)
            ig_attributes.append(gain)
        return attributes[np.argmax(ig_attributes)]

    def find_entropy(self, df):
        """
        Find total entropy from an attribute
        """
        Class = df.keys()[-1]
        entropy = 0
        labels = df[Class].unique()
        for label in labels:
            fraction = df[Class].value_counts()[label]/len(df[Class])
            entropy += -fraction*np.log2(fraction)
        return entropy

    def find_entropy_attribute(self, df, attribute):
        """
        Find entropy from different values of an attribute
        """
        eps = np.finfo(float).eps
        # Class is target variable
        Class = df.keys()[-1]
        # labels are values in target variable (like 'YES'/'NO' or 0/1 etc.)
        labels = df[Class].unique()
        # variables are different values in that attribute (like 'Hot','Cold' in Temperature attribute)
        variables = df[attribute].unique()
        entropy2 = 0
        for variable in variables:
            entropy = 0
            for label in labels:
                numer = len(df[attribute][df[attribute]==variable][df[Class] ==label])
                denom = len(df[attribute][df[attribute]==variable])
                fraction = numer/(denom+eps)
                entropy += -fraction*np.log2(fraction+eps)
            fraction2 = denom/len(df)
            entropy2 += -fraction2*entropy
        return abs(entropy2)

    def get_subtable(self, df, node, value):
        """
        Getting subset of dataset
        """
        return df[df[node] == value].reset_index(drop=True)

    def print_tree(self):
        """
        Print the decision tree
        """
        print('\n\t\t\t\t===Decision Tree===\n')
        pprint(self.tree)
