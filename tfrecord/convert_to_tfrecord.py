# @Author:hemin.wang
import numpy as np

import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_float_dtype
from pandas.api.types import is_integer_dtype
import math
import tensorflow as tf

def _create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _create_string_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def get_df_from_csv(input_csv_file_path):
    """
    读取csv文件生成pandas dataframe，缺失值string列由other填充，float列由0.0填充，int列由0填充
    :param input_csv_file_path:
    :return:没有缺失值的完整dataframe
    """
    def convert_to_float(value):
        res=0.0
        # 判断是否为float64的nan值
        if math.isnan(value):
            return res
        try:
            res=float(value)
        except:
            pass
        return res

    def convert_to_int(value):
        res=0
        try:
            res=int(value)
        except:
            pass
        return res

    def convert_to_string(value):
        res="other"
        try:
            res=str(value)
        except:
            pass
        return res

    df=pd.read_csv(input_csv_file_path)
    for col in df.columns:
        if is_string_dtype(df[col]):
            df[col].fillna("other",inplace=True)
            df[col]=df[col].map(convert_to_string)
        elif is_float_dtype(df[col]):
            df[col].fillna(0.0,inplace=True)
            df[col]=df[col].map(convert_to_float)
            # df[col]=df[col].astype(np.float32)   默认pandas Dataframe的float数据类型为float64，而tfrecord读取float32，目前看此行有无均可，待查阅
        elif is_integer_dtype(df[col]):
            df[col].fillna(0,inplace=True)
            df[col]=df[col].map(convert_to_int)
    print(df.info())
    return df


def write_df_to_tfrecord(input_csv_file_path,output_tfrecord_file_path):
    """
    将从csv文件读取的pandas dataframe写入tfRecord
    :param input_csv_file_path:
    :param output_tfrecord_file_path:
    :return:
    """
    writer = tf.io.TFRecordWriter(output_tfrecord_file_path)
    df=get_df_from_csv(input_csv_file_path)
    counter=0
    for i in range(df.shape[0]):
        feed_dict = {}
        for col in df.columns:
            if is_string_dtype(df[col]):
                feed_dict.update({col:_create_string_feature([df[col].iloc[i].encode()])})
            elif is_float_dtype(df[col]):
                feed_dict.update({col:_create_float_feature([df[col].iloc[i]])})
            elif is_integer_dtype(df[col]):
                feed_dict.update({col:_create_int_feature([df[col].iloc[i]])})
        counter += 1
        example=tf.train.Example(features=tf.train.Features(feature=feed_dict))
        serialized=example.SerializeToString()
        writer.write(serialized)

def get_df_col_schema(input_csv_file_path,output_schema_file_path):
    """
    通过读取pandas dataframe的每列信息，获取解析其生成的TFRecord时所需要的schema信息，"字段名，字段数据类型，字段值长度"格式
    :param input_csv_file_path:
    :param output_schema_file_path:
    :return:
    """
    df = get_df_from_csv(input_csv_file_path)
    with open(output_schema_file_path,"w",encoding="utf-8") as fw:
        for col in df.columns:Data columns
            if is_string_dtype(df[col]):
                fw.write(col+",string"+",1"+"\n")
            elif is_float_dtype(df[col]):
                fw.write(col+",float"+",1"+"\n")
            elif is_integer_dtype(df[col]):
                fw.write(col+",int"+",1"+"\n")

if __name__ == '__main__':
    # df=get_df_from_csv("datas/testSamples.csv")

    write_df_to_tfrecord("datas/testSamples.csv","datas/test_sample.tfrecord")
    get_df_col_schema("datas/testSamples.csv","datas/test_sample.tfrecord.schema")

