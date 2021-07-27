# @Author:hemin.wang
from absl import logging
import tensorflow as tf

class TFRecordParser():
    def __init__(self,input_tfrecord_file_path,
                 tfrecord_schema_file_path,
                 batch_size=32,
                 repeat=False,shuffle=False,drop_remainder=False):
        self.input_tfrecord_file_path=input_tfrecord_file_path
        self.tfrecord_schema_file_path=tfrecord_schema_file_path
        self.features_dict=self.get_parse_feature_dict()
        self.batch_size=batch_size
        self.repeat = repeat
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder

    def get_feature_schema_dict(self):
        feature_schema_dict = {}
        input = open(self.tfrecord_schema_file_path, "r", encoding="utf-8")
        line = str(input.readline())
        while line != None and len(line) > 1:
            line_split = line.strip().split(",")
            if len(line_split) != 3:
                logging.info(
                    "in the schema file {input_schema_file_path},the schema of feature {feature_name} is invalid"
                    .format(input_schema_file_path=self.tfrecord_schema_file_path, feature_name=line_split[0]))
            feature_name = line_split[0]
            feature_type = line_split[1]
            feature_length = int(line_split[2])
            if feature_name in feature_schema_dict:
                logging.debug("in the schema file {input_schema_file_path},the feature {feature_name} is duplicate"
                              .format(input_schema_file_path=self.tfrecord_schema_file_path, feature_name=feature_name))
            else:
                feature_schema_dict.update({feature_name: {"type": feature_type, "length": feature_length}})
            line = str(input.readline())
        return feature_schema_dict

    def get_parse_feature_dict(self):
        feature_schema_dict=self.get_feature_schema_dict()
        features_dict = dict()
        for k, v in feature_schema_dict.items():
            if v.get("type") == "string":
                features_dict.update({k: tf.io.FixedLenFeature([v.get("length", 1)], tf.string)})
            elif v.get("type") == "float":
                features_dict.update({k: tf.io.FixedLenFeature([v.get("length", 1)], tf.float32)})
            elif v.get("type") == "int":
                features_dict.update({k: tf.io.FixedLenFeature([v.get("length", 1)], tf.int64)})
        return features_dict

    def parse_tf_example(self,example):
        parsed_example=tf.io.parse_single_example(example,self.features_dict)
        return parsed_example


    def parse_tfrecord(self):
        files=tf.io.gfile.glob(self.input_tfrecord_file_path)
        data_set=tf.data.TFRecordDataset(files)
        data_set=data_set.map(self.parse_tf_example)
        if self.shuffle:
            data_set=data_set.shuffle(self.batch_size)
        if self.repeat:
            data_set.repeat()
        data_set=data_set.batch(self.batch_size,drop_remainder=self.drop_remainder)
        for data in data_set:
            yield data


def _test_parse_tfrecord():
    print(tf_record_parser.features_dict)
    counter=0
    for data in tf_record_parser.parse_tfrecord():
        if counter>1:
            break
        print(data)
        print(data["label"])
        counter+=1


if __name__ == '__main__':
    tf_record_parser=TFRecordParser(input_tfrecord_file_path="datas/test_sample.tfrecord",
                                    tfrecord_schema_file_path="datas/test_sample.tfrecord.schema",
                                    batch_size=2)
    _test_parse_tfrecord()
