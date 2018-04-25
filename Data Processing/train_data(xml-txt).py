# -*- coding: UTF-8 -*-
import xml.sax
import os
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser


#导入模型
LTP_DATA_DIR = '/Users/zhouyujie/NLP_source/pyltp/ltp_data'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 句法分析模型路径，模型名称为`parser.model`


segmentor = Segmentor()  # 初始化分词实例
segmentor.load(cws_model_path)  # 加载分词模型

postagger = Postagger() # 初始化标注实例
postagger.load(pos_model_path)  # 加载标注模型

parser = Parser() # 初始化实例
parser.load(par_model_path)  # 加载模型


class DataHandler( xml.sax.ContentHandler ):

   def __init__(self):
      self.CurrentData = ""
      self.TEXT = ""
      self.CORRECTION = ""
      self.ERROR = ""
      self.end_off='-1'
      self.start_off='-1'
      self.E_type=""
      self.start=[]
      self.end=[]
      self.type=[]
      self.word=[]
      self.tag=[]
      self.acs=[]


   # 元素开始事件处理
   def startElement(self, tag, attributes):
      self.CurrentData = tag
      if self.CurrentData == "DOC":
         print(self.word)
         print(self.tag)
         arcs=parser.parse(self.word, self.tag)# 句法分析
         parserlist=[]
         for arc in arcs:
            parserlist.append(str(arc.head))
            #parserlist.append(str(arc.head) + ":" + arc.relation)
         print(parserlist)
         n = 0
         label = []
         for i in range(len(self.word)):
            for j in range(len(self.word[i])):
               label.append('O')
               n = n + 1
               for x in range(len(self.start)):
                  if n == int(self.start[x]):
                     if self.type[x] == 'S':
                        label[n - 1] = 'B-S'
                     elif self.type[x] == 'R':
                        label[n - 1] = 'B-R'
                     elif self.type[x] == 'W':
                        label[n - 1] = 'B-W'
                     elif self.type[x] == 'M':
                        label[n - 1] = 'B-M'
                  elif n > int(self.start[x]) and n <= int(self.end[x]) and label[n - 1] != 'B':
                     if self.type[x] == 'S':
                        label[n - 1] = 'I-S'
                     elif self.type[x] == 'R':
                        label[n - 1] = 'I-R'
                     elif self.type[x] == 'W':
                        label[n - 1] = 'I-W'
                     elif self.type[x] == 'M':
                        label[n - 1] = 'I-M'
               if j == 0:
                  #f.writelines([self.word[i][j], ' ', 'B-' + self.tag[i], ' ', parserlist[i],' ', label[n - 1], '\n'])
                  f.writelines([self.word[i][j], ' ', 'B-' + self.tag[i], ' ', label[n - 1], '\n'])
               else:
                  #f.writelines([self.word[i][j], ' ', 'I-' + self.tag[i], ' ', parserlist[i], ' ',label[n - 1],'\n'])
                  f.writelines([self.word[i][j], ' ', 'B-' + self.tag[i], ' ', label[n - 1], '\n'])
         self.start=[]
         self.end=[]
         self.type=[]
         f.write('\n')
      if self.CurrentData == "ERROR":
         self.end_off = attributes["end_off"]
         self.start_off = attributes["start_off"]
         self.E_type=attributes["type"]
         self.start.append(self.start_off)
         self.end.append(self.end_off)
         self.type.append(self.E_type)
         #print("ERROR",self.end_off,',',self.start_off,',',self.E_type)
      word_list = segmentor.segment(self.TEXT)  # 分词
      tag_list = postagger.postag(word_list)  # 词性标注
      self.word = list(word_list)
      self.tag = list(tag_list)

   def endElement(self, tag):
      if self.CurrentData == "TEXT":
         print("TEXT:", self.TEXT)
      elif self.CurrentData == "CORRECTION":
         print("CORRECTION:", self.CORRECTION)
      self.CurrentData = ""


   # 内容事件处理
   def characters(self, content):
      if self.CurrentData == "TEXT":
         self.TEXT = content
      elif self.CurrentData == "CORRECTION":
         self.CORRECTION = content
      elif self.CurrentData == "ERROR":
         self.ERROR= content




f = open('/Users/zhouyujie/Desktop/CRF-output.txt','w')

if (__name__ == "__main__"):
   # 创建一个 XMLReader
   myparser = xml.sax.make_parser()
   # turn off namepsaces
   myparser.setFeature(xml.sax.handler.feature_namespaces, 0)
   # 重写 ContextHandler
   Handler = DataHandler()
   myparser.setContentHandler(Handler)
   myparser.parse("/Users/zhouyujie/Desktop/CGED16_HSK_TrainingSet.xml")


