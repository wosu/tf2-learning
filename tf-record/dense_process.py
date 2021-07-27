# @Author:hemin.wang

import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))

import numpy as np
import tensorflow as tf
from tfrecord.parse_tfrecord_data import TFRecordParser

class EmbeddingEncoder():
    """
    根据是否需要加载预训练embedding，将id类特征转换为embedding
    """

    def get_embedding_feature_schema_dict(self,input_feature_schema_file_path):
        """
        从input_feature.schema文件中解析出需要进行embedding的feature及其相关参数,文件格式为（特征名，特征转换类型，特征取值数，embedding后的维度，预训练文件路径）
        :param input_feature_schema_file_path:
        :return:{'movieId': {'feature_cate_num': 1001, 'output_dim': 16},
        'userId': {'feature_cate_num': 30001, 'output_dim': 32, 'pre_embedding_file_path': 'datas/test_pre_embedding'}}
        """
        embedding_schema_dict = {}
        input = open(input_feature_schema_file_path, "r", encoding="utf-8")
        line = str(input.readline())
        while line != None and len(line) > 1:
            line_split = line.strip().split(",")
            input_type = line_split[1]
            if input_type == "embedding":
                feature_name = line_split[0]
                feature_cate_num = int(line_split[2])
                output_dim = int(line_split[3])
                if len(line_split) >= 5:
                    pre_embedding_file=line_split[4]
                    embedding_schema_dict.update(
                        {feature_name: {"feature_cate_num": feature_cate_num,
                                        "output_dim": output_dim,
                                        "pre_embedding_file_path":pre_embedding_file}})
                else:
                    embedding_schema_dict.update(
                        {feature_name: {"feature_cate_num": feature_cate_num, "output_dim": output_dim}})

            line = str(input.readline())
        return embedding_schema_dict

    def get_embedding_layer_dicts(self, input_feature_schema_file_path):
        """
        通过读取input_feature_schema_file_path中需要embedding的特征的配置，生成模型中需要的所有embedding层组成的字典
        :param input_feature_schema_file_path:
        :return: {'movieId_embedding_layer': <tensorflow.python.keras.layers.embeddings.Embedding object at 0x102934880>,
        'userId_embedding_layer': <tensorflow.python.keras.layers.embeddings.Embedding object at 0x11b623f10>}
        """
        embedding_layer_dict = {}
        for feature, v in self.get_embedding_feature_schema_dict(input_feature_schema_file_path).items():
            if v.get("pre_embedding_file_path") is not None:
                embedding_layer = self.build_embedding_layer(feature_name=feature,
                                                             output_dim=v.get("output_dim"),
                                                             pre_embedding_file_path=v.get(
                                                                 "pre_embedding_file_path"))

                embedding_layer_dict.update({embedding_layer.name:embedding_layer})
            else:
                embedding_layer = self.build_embedding_layer(feature_name=feature,
                                                             feature_cate_num=v.get("feature_cate_num"),
                                                             output_dim=v.get("output_dim"))
                embedding_layer_dict.update({embedding_layer.name:embedding_layer})
        return embedding_layer_dict

    def get_pre_embedding_matrix(self,pre_embedding_file_path):
        """
        对于预训练的embedding，需从保存预训练结果的pre_embedding_file文件中读取出预训练的embedding_matrix
        :param pre_embedding_file_path:
        :return:
        """
        embedding_matrix=[]
        input=open(pre_embedding_file_path,"r",encoding="utf-8")
        line=str(input.readline())
        while line!=None and len(line)>1:
            id,embedding=line.strip().split("\t")
            embedding_matrix.append(eval(embedding))
            line=str(input.readline())
        embedding_0=[0]*len(embedding_matrix[0])
        embedding_matrix=[embedding_0]+embedding_matrix
        embedding_matrix=np.array(embedding_matrix)
        return embedding_matrix

    def build_embedding_layer(self,
                              feature_name,
                              feature_cate_num=10,
                              # input_length=1,
                              output_dim=32,
                              trainable=True,pre_embedding_file_path=None):
        """
        构建embedding_layer.根据是否有预训练embedding文件判断是使用预训练embedding
        :param feature_name: 需要做embedding转换的特征名
        :param feature_cate_num: 特征的原始取值个数
        :param output_dim: 输出的embedding维数
        :param trainable:是否可训练，默认为可训练
        :param pre_embedding_file_path:预训练embedding文件位置
        :return:DNN的embedding层
        """
        if pre_embedding_file_path is not None:
            embedding_matrix=self.get_pre_embedding_matrix(pre_embedding_file_path)
            embedding_layer=tf.keras.layers.Embedding(input_dim=len(embedding_matrix),
                                                      output_dim=len(embedding_matrix[0]),
                                                      weights=[embedding_matrix],
                                                      # input_length=input_length,
                                                      trainable=trainable,
                                                      name=feature_name+"_embedding_layer")
        else:
            embedding_layer=tf.keras.layers.Embedding(input_dim=feature_cate_num,
                                                      output_dim=output_dim,
                                                      # input_length=input_length,
                                                      trainable=trainable,
                                                      name=feature_name+"_embedding_layer")
        return embedding_layer

def _test_embedding_matrix():
    embedding_matrix = embedding_encoder.get_pre_embedding_matrix("datas/test_pre_embedding")
    print(embedding_matrix)
    print(len(embedding_matrix))

def _test_embedding_layer():
    embedding_layer = embedding_encoder.build_embedding_layer(feature_name="userId",
                                                              feature_cate_num=10,
                                                              input_length=1,
                                                              output_dim=5,
                                                              trainable=True)
    print(embedding_layer)
    print(embedding_layer.name)
    print(type(embedding_layer.name))
    print(embedding_layer.weights)
    print(embedding_layer.trainable)
    print(embedding_layer.input_length)


def _test_pre_embedding_layer():
    embeding_layer=embedding_encoder.build_embedding_layer(feature_name="userId",
                                                           trainable=False,
                                                           pre_embedding_file_path="datas/test_pre_embedding")
    print(embeding_layer)
    print(embeding_layer.name)
    print(embeding_layer.weights)
    print(embeding_layer.trainable)
    print(embeding_layer.output_dim)
    print(embeding_layer.input_length)

def _test_tfrecord_embedding_layer():
    userId_inputs = tf.keras.Input(shape=(30,), name="userId_inputs")
    userId_embeding_layer = embedding_encoder.build_embedding_layer(feature_name="userId",
                                                             trainable=False,
                                                             pre_embedding_file_path="datas/test_pre_embedding")
    print(userId_embeding_layer.weights)
    userId_embedding=userId_embeding_layer(userId_inputs)
    print(userId_embedding)

    tfrecord_parser = TFRecordParser(input_tfrecord_file_path="../tfrecord/datas/test_sample.tfrecord",
                                     tfrecord_schema_file_path="../tfrecord/datas/test_sample.tfrecord.schema",
                                     batch_size=1)
    # counter = 0
    # for data in tfrecord_parser.parse_tfrecord():
    #     print(counter,"***************************************")
    #     if counter > 1:
    #         break
    #     print(data)
    #     counter+=1

def _test_get_embedding_feature_schema_dict():
    embedding_schema_dict=embedding_encoder.get_embedding_feature_schema_dict("datas/input_feature.schema")
    print(embedding_schema_dict)


def _test_get_embedding_layer_dicts():
    embedding_layer_dict=embedding_encoder.get_embedding_layer_dicts("datas/input_feature.schema")
    print(embedding_layer_dict)



if __name__ == '__main__':
    embedding_encoder=EmbeddingEncoder()
    # _test_embedding_matrix()
    # _test_embedding_layer()
    # _test_pre_embedding_layer()
    # _test_tfrecord_embedding_layer()
    _test_get_embedding_feature_schema_dict()
    _test_get_embedding_layer_dicts()



