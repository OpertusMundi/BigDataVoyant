import csv
import string
import typing
from itertools import islice

import numpy as np
from scipy import stats


import phonenumbers
import re
from dateutil.parser import parse


def is_phone(column, top_rows_to_search=10000):
    row_counter = 0
    for row in column:
        try:
            z = phonenumbers.parse(str(row), None)
        except phonenumbers.phonenumberutil.NumberParseException:
            pass
        else:
            if phonenumbers.is_valid_number(z):
                return True
        row_counter += 1
        if row_counter == top_rows_to_search:
            return False
    return False


def is_email(column, top_rows_to_search=10000):
    regex = re.compile(
        r"(^[-!#$%&'*+/=?^_`{}|~0-9A-Z]+(\.[-!#$%&'*+/=?^_`{}|~0-9A-Z]+)*"  # dot-atom
        r'|^"([\001-\010\013\014\016-\037!#-\[\]-\177]|\\[\001-\011\013\014\016-\177])*"'
        r')@(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?$', re.IGNORECASE)  # domain
    row_counter = 0
    for row in column:
        if regex.match(str(row)) is not None:
            return True
        row_counter += 1
        if row_counter == top_rows_to_search:
            return False
    return False


def is_url(column, top_rows_to_search=10000):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    row_counter = 0
    for row in column:
        if regex.match(str(row)) is not None:
            return True
        row_counter += 1
        if row_counter == top_rows_to_search:
            return False
    return False


def is_image_url(column, top_rows_to_search=10000):
    image_file_endings = ('.apng', '.avif', '.gif', '.jpg', '.jpeg', '.jfif', '.pjpeg', '.pjp', '.png', '.svg',
                          '.webp', '.bmp', '.ico', '.cur', '.tif', '.tiff')
    row_counter = 0
    for row in column:
        if str(row).endswith(image_file_endings):
            return True
        row_counter += 1
        if row_counter == top_rows_to_search:
            return False
    return False


def is_datetime(column, top_rows_to_search=1000):
    row_counter = 0
    for row in column:
        try:
            parsed_date = parse(str(row))
            if not (parsed_date.month == 1 and parsed_date.day == 31
                    and parsed_date.hour + parsed_date.minute + parsed_date.second == 0):
                return True
        except ValueError:
            return False
        except OverflowError:
            return False;
        row_counter += 1
        if row_counter == top_rows_to_search:
            return False
    return False


def keywords_per_column(column, top_n=3):
    frequencies = {}
    top_n_items = []
    for row in column:
        if isinstance(row, str):
            row = row.translate(str.maketrans('', '', string.punctuation)).split(" ")
            for word in row:
                if word:
                    frequencies[word] = frequencies.get(word, 0) + 1
            frequencies = {k: v for k, v in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}
            top_n_items = list(islice(frequencies.items(), top_n))
    return top_n_items


def get_delimiter(str_value):
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(str_value, ['\t', '|', ';', ' '])
        return dialect.delimiter
    except csv.Error:
        return ''


def get_decimal_point(row, detected_delimiter):
    if detected_delimiter == '' and row.count(',') > 1:
        detected_delimiter = ','
        items = row.split(detected_delimiter)
        item_for_cmp = items[0]
    else:
        items = row.split(detected_delimiter) if detected_delimiter != '' else [row]
        item_for_cmp = items[0]
    if ',' in item_for_cmp:
        float(item_for_cmp.replace(',', '.'))
        return 'floats with , as the decimal point', detected_delimiter
    elif '.' in item_for_cmp:
        float(item_for_cmp)
        return 'floats with . as the decimal point', detected_delimiter
    else:
        int(item_for_cmp)
        return 'integers', detected_delimiter


def numerical_value_pattern(column):
    for row in column:
        row = str(row)
        if row.strip() != '':
            detected_delimiter = get_delimiter(row)
            try:
                decimal_point_str_value, detected_delimiter = get_decimal_point(row, detected_delimiter)
                if detected_delimiter == '':
                    return f'Column contains rows with single {decimal_point_str_value}'
                else:
                    if detected_delimiter == '\t':
                        detected_delimiter = 'TAB'
                    elif detected_delimiter == ' ':
                        detected_delimiter = 'single whitespace'
                    return f'Column contains rows  with multiple values that are of type {decimal_point_str_value} ' \
                           f'delimited with {detected_delimiter}'
            except ValueError:
                return ''


def numerical_statistics(column):
    try:
        modal_num, count = stats.mode(column)
        modal_value = {"value": modal_num[0], "count": count[0]}
    except IndexError:
        modal_value = None
    try:
        ptp = np.ptp(column)
    except ValueError:
        ptp = None
    return {
        "median": np.median(column),
        "mean": np.mean(column),
        "variance": np.var(column),
        "stdev": np.std(column),
        "peak-to-peak range": ptp,
        "modal value": modal_value,
    }


def transform_column_to_float(column):
    try:
        column = column.astype(np.float)
        column = column[~np.isnan(column)]
        if np.isnan(column).all():
            return None
        if np.all(column == column[0]):
            return None
    except ValueError:
        return None
    else:
        return column


def get_shortest_list_len(lst):
    return len(min(lst, key=len))


def make_square_2d_list(lst_2d, size):
    return [lst_1d[:size] for lst_1d in lst_2d]


def correlation_among_numerical_attributes(numerical_columns):
    true_numerical_columns = []
    true_numerical_column_names = []
    for col_name, numerical_column in numerical_columns:
        transformed_column = transform_column_to_float(numerical_column)
        if transformed_column is not None:
            true_numerical_columns.append(transformed_column)
            true_numerical_column_names.append(col_name)
    s_len = get_shortest_list_len(true_numerical_columns)
    sq_true_numerical_columns = make_square_2d_list(true_numerical_columns, s_len)
    corr = np.corrcoef(sq_true_numerical_columns).tolist()
    return true_numerical_column_names, corr


def histogram(numerical_column):
    bucket_counts, bucket_ranges = np.histogram(np.isfinite(numerical_column), bins='auto')
    return list(bucket_counts), list(bucket_ranges)


def date_time_value_distribution(column):
    if not is_datetime(column):
        return None
    dates = []
    for date in column:
        try:
            parsed_date = parse(str(date))
            dates.append(f'{parsed_date.month}/{parsed_date.year}')
        except ValueError:
            return None
    y, count = np.unique(dates, return_counts=True)
    distribution = {y[i]: count[i] for i in range(len(y))}
    return distribution


def uniqueness(column):
    column = np.array([hash(i) for i in column if i], dtype=np.uint64)
    total = len(column)
    if total == 0:
        return None
    n_unique = len(np.unique(column))
    return n_unique / total
