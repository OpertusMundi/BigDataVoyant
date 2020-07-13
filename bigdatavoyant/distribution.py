import json
import pandas as pd
import matplotlib.pyplot as plt

class Distribution(object):
    def __init__(self, df, attributes, n_obs, dropmissing=True, method='brute'):
        all_attributes = df.get_column_names(virtual=False)
        if (attributes is None):
            attributes = all_attributes
        else:
            if not isinstance(attributes, list):
                attributes = [attributes]
            for attribute in attributes:
                if (attribute not in all_attributes):
                    raise Exception('Attribute %s not found' % (attribute))
        distribution = {}
        self._attributes = attributes
        if method == 'brute':
            for column in attributes:
                value_counts = df[column].value_counts(dropmissing=dropmissing)
                sum_all = value_counts.sum()
                value_counts = value_counts.head(n_obs)
                sum_head = value_counts.sum()
                sum_other = sum_all - sum_head
                value_counts = json.loads(value_counts.to_json())
                distribution[column] = value_counts
        elif method == 'ml':
            import vaex.ml
            df = df.to_vaex_df()
            for column in attributes:
                encoder = vaex.ml.FrequencyEncoder(features=[column])
                cdf = df[[column]]
                if dropmissing:
                    cdf = cdf.dropna()
                cdf.ordinal_encode(column, inplace=True)
                fit = encoder.fit_transform(cdf)
                unique = fit[[column, 'frequency_encoded_{}'.format(column)]].unique(column)
                value_counts = fit.groupby(column, 'min').sort('frequency_encoded_{}_min'.format(column), ascending=False)
                count = df.count(column)
                value_counts['count'] = value_counts.apply(lambda x: int(x*count), arguments=[value_counts['frequency_encoded_{}_min'.format(column)]])
                value_counts = value_counts.head(n_obs).to_dict()
                value_counts = dict(zip(value_counts[column], value_counts['count']))
                distribution[column] = value_counts
        else:
            raise Exception('Method could be one of "brute" or "ml", %s not supported.' % (method))
        self._distribution = distribution

    def to_dict(self):
        return self._distribution

    def to_json(self):
        return json.dumps(self.to_dict())

    def plot(self):
        length = len(self._attributes)
        fig, axs = plt.subplots(length, figsize=(10,5*length))
        i=0
        for column in self._attributes:
            value_counts = self._distribution[column]
            if (length > 1):
                axs[i].bar(list(value_counts), list(value_counts.values()))
                axs[i].set(title=column, ylabel='freq')
            else:
                axs.bar(list(value_counts), list(value_counts.values()))
                axs.set(title=column, ylabel='freq')
            i+=1
        plt.show()
