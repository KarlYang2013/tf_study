#coding:utf-8
#加上正则
import tensorflow as tf
#用tf.data.TextLineDataset从CSV读取数据，然后创建dataset
#data来自：https://www.kaggle.com/c/GiveMeSomeCredit/data
import tensorflow as tf
import numpy as np

_CSV_COLUMNS =  [u'Unnamed: 0', u'DebtRatio', u'MonthlyIncome', u'NumberOfDependents',
                 u'NumberOfOpenCreditLinesAndLoans',
                   u'NumberOfTime30-59DaysPastDueNotWorse',
                   u'NumberOfTime60-89DaysPastDueNotWorse', u'NumberOfTimes90DaysLate',
                   u'NumberRealEstateLoansOrLines',
                   u'RevolvingUtilizationOfUnsecuredLines', u'age', u'SeriousDlqin2yrs']
_CSV_COLUMN_DEFAULTS=12*[0.0] #每列的默认值

def input_fn(data_file, shuffle, batch_size):
  def parse_csv(value):
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, field_delim=',')
    features = dict(zip(_CSV_COLUMNS, columns))
    #labels = features.pop(u'SeriousDlqin2yrs')
    #eturn features, tf.equal(labels, 1.0)
    return features

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)
  if shuffle: dataset = dataset.shuffle(buffer_size=100000)
  dataset = dataset.map(parse_csv, num_parallel_calls=100)
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  return dataset

BATCH_SIZE = 2
SHUFFLE_FLAG = True
data = input_fn("train_feature.csv", shuffle = SHUFFLE_FLAG, batch_size = BATCH_SIZE)
iter = data.make_one_shot_iterator()

with tf.Session() as sess:
    sample = sess.run(iter.get_next()) #解析数据
    print(sample)
    #打印batch里的每一个样本
    for idx in range(BATCH_SIZE):
        print("the %d-th sample:"%(idx))
        for col in _CSV_COLUMNS:
            print(sample[col][idx])
