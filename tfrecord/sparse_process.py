# @Author:hemin.wang
import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
from tfrecord.parse_tfrecord_data import TFRecordParser
import numpy as np

class OnehotEncoder():
    def __init__(self,input_tfrecord_file_path,tfrecord_schema_file_path,input_feature_schema_file_path,batch_size=32):
        """
        将解析出来的tfrecord中需要做onehot的特征进行onehot变换，以迭代器的方式输出每个batch中指定特征的onehot结果，或所有onehot特征组成的列表结果
        :param input_tfrecord_file_path:需要解析的tfrecord文件
        :param tfrecord_schema_file_path:tfrecord文件的字段schema文件
        :param input_feature_schema_file_path:需作为模型输入的特征schema
        :param batch_size:
        """
        self.input_tfrecord_file_path=input_tfrecord_file_path
        self.tfrecord_schema_file_path=tfrecord_schema_file_path
        self.input_feature_schema_file_path=input_feature_schema_file_path
        self.onehot_inputs_schema_dict=self.get_onehot_inputs_schema()
        self.batch_size=batch_size


    def get_onehot_inputs_schema(self):
        """
        从input_feature.schema文件中解析出需要进行onehot为input的feature及其相关参数,文件格式为（特征名，特征转换类型，特征取值数，起始索引(0或1)）
        :return:
        """
        onehot_inputs_schema_dict = {}
        input = open(self.input_feature_schema_file_path, "r", encoding="utf-8")
        line = str(input.readline())
        while line != None and len(line) > 1:
            line_split = line.strip().split(",")
            input_type = line_split[1]
            if input_type == "onehot":
                feature_name = line_split[0]
                onehot_dim_size = int(line_split[2])
                start_index = int(line_split[3])
                onehot_inputs_schema_dict.update(
                    {feature_name: {"onehot_dim_size": onehot_dim_size, "start_index": start_index}})
            line = str(input.readline())
        return onehot_inputs_schema_dict

    def tfrecord_onefeature_encode_onehot_input(self,feature_name,dim_size,start_index=0):
        """
        将tfrecord解析出来的指定特征（feature_name）转换为onehot，转换后的onehot维度为dim_size参数值，默认该特征起始索引为0
        :param feature_name:
        :param dim_size:
        :param start_index:
        :return:
        """
        tfrecord_parser = TFRecordParser(input_tfrecord_file_path=self.input_tfrecord_file_path,
                                         tfrecord_schema_file_path=self.tfrecord_schema_file_path,
                                         batch_size=self.batch_size)
        for data in tfrecord_parser.parse_tfrecord():
            onehot_res=onehot_encode(data[feature_name],dim_size=dim_size,start_index=start_index)
            yield onehot_res


    def tfrecord_encode_onehot_inputs(self):
        """
        根据需转onehot的schema，将tfrecord解析出的所有需做onehot转换的字段转换为onehot，返回所有转换后的所有onehot组成的列表
        :return:
        """
        tfrecord_parser = TFRecordParser(input_tfrecord_file_path=self.input_tfrecord_file_path,
                                         tfrecord_schema_file_path=self.tfrecord_schema_file_path,
                                         batch_size=self.batch_size)

        for data in tfrecord_parser.parse_tfrecord():
            onehot_res=[]
            for k,v in self.onehot_inputs_schema_dict.items():
                onehot_res.append(onehot_encode(data[k],dim_size=v.get("onehot_dim_size",1),start_index=v.get("start_index",0)))
            yield onehot_res



def onehot_encode(values,dim_size,start_index=0):
    """
    将id索引类特征转换为onehot特征，分两种情况，id从0开始还是id从1开始
    :param values:需要转换的原始id类特征，一般为tensor格式的二维数组
    :param dim_size:最终生成的onehot的维度
    :param start_index:id索引从0开始
    :return:二维数组格式的onehot结果
    """
    res_values = []
    for row in values:
        res_row = [0] * dim_size
        for value in row:
            if start_index == 0:
                if 0 <= value < dim_size:
                    res_row[value] = 1
            else:
                if 0 < value <= dim_size:
                    res_row[value - 1] = 1
        res_values.append(res_row)
    return np.array(res_values)



def _test_onehot_encode():
    res=onehot_encode(np.array([[1],[2]]),dim_size=3,start_index=0)
    print(res)



def _test_tfrecord_onefeature_encode_onehot_input():
    onehots=onehotEncoder.tfrecord_onefeature_encode_onehot_input(feature_name="movieId",dim_size=1001,start_index=1)
    counter=0
    for onehot in onehots:
        if counter>1:
            break
        print(onehot)
        counter+=1



def _test_tfrecord_encode_onehot_inputs():
    onehot_list_generator=onehotEncoder.tfrecord_encode_onehot_inputs()
    counter=0
    for onehot_list in onehot_list_generator:
        # if counter>1:
        #     break
        print(onehot_list)
        counter+=1

if __name__ == '__main__':
    tfrecordParser = TFRecordParser(input_tfrecord_file_path="../tfrecord/datas/test_sample.tfrecord",
                                    tfrecord_schema_file_path="../tfrecord/datas/test_sample.tfrecord.schema",
                                    batch_size=2)

    onehotEncoder=OnehotEncoder(input_tfrecord_file_path="../tfrecord/datas/test_sample.tfrecord",
                                tfrecord_schema_file_path="../tfrecord/datas/test_sample.tfrecord.schema",
                                input_feature_schema_file_path="datas/input_feature.schema",
                                batch_size=2)
    #test_onehot_encode()
    # _test_tfrecord_onefeature_encode_onehot_input()
    _test_tfrecord_encode_onehot_inputs()
