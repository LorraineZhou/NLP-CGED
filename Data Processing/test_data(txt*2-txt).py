# -*- coding: UTF-8 -*-
import xml.sax
import os
import sys
from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser

input=[]
truth=[]
article=''
word_list=[]
tag_list=[]
input_txt=open('CGED16_HSK_TEST_Input.txt').readlines()
truth_txt=open('CGED16_HSK_TEST_Truth.txt').readlines()

for i in range(len(input_txt)):
    input_txt[i]= input_txt[i].replace('\n', ' ');
    input.append(input_txt[i].split('\t'))



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



f = open("CGED16_Input.txt",'w')

for i in range(len(input)):
    flag=0
    for j in range(len(truth_txt)):
        truth_txt[j] = truth_txt[j].replace(' ', '');
        truth_txt[j] = truth_txt[j].replace('\n', '');
        truth.append(truth_txt[j].split(','))
        word_list = segmentor.segment(input[i][1])  # 分词
        tag_list = postagger.postag(word_list)  # 词性标注
        word = list(word_list)
        tag = list(tag_list)
        num=0;

        if input[i][0].find(truth[j][0]) != -1:
            if truth[j][1] == 'correct':
                lines = []
                for m in range(len(word)):
                    for n in range(len(word[m])):
                        num=num+1;
                        if n == 0:
                            lines.append(word[m][n]+'\t'+'B-' + tag[m]+'\t'+ 'O'+ '\t'+'\n');
                            # f.writelines([word[m][n], '\t', 'B-' + tag[m], '\t', '0', '\t', '\n'])
                        else:
                            lines.append(word[m][n] + '\t' + 'I-' + tag[m] + '\t' + 'O' + '\t' + '\n');
                            # f.writelines([word[m][n], '\t', 'I-' + tag[m], '\t', '0', '\t', '\n'])
                f.writelines(lines)
            else:
                # print(truth[j])
                if flag==0:
                    flag = flag + 1
                    lines = []
                    for m in range(len(word)):
                        for n in range(len(word[m])):
                            num=num+1
                            if int(truth[j][1]) == num:
                                if n == 0:
                                    lines.append(word[m][n] + '\t' + 'B-' + tag[m] + '\t' + 'B-'+truth[j][3]+ '\t' + '\n')
                                    #f.writelines([word[m][n], '\t', 'B-' + tag[m], '\t', 'B-'+truth[j][3], '\t','\n'])
                                else:
                                    lines.append(word[m][n] + '\t' + 'I-' + tag[m] + '\t' + 'B-' + truth[j][3] + '\t' + '\n')
                                    # f.writelines([word[m][n], '\t', 'I-' + tag[m], '\t', 'B-'+truth[j][3], '\t', '\n'])
                            elif (int(truth[j][2]) >= num) and (int(truth[j][1]) < num):
                                if n == 0:
                                    lines.append(word[m][n] + '\t' + 'B-' + tag[m] + '\t' + 'I-' + truth[j][3] + '\t' + '\n')
                                    # f.writelines([word[m][n], '\t', 'B-' + tag[m], '\t', 'I-' + truth[j][3], '\t', '\n'])
                                else:
                                    lines.append(word[m][n] + '\t' + 'I-' + tag[m] + '\t' + 'I-' + truth[j][3] + '\t' + '\n')
                                    # f.writelines([word[m][n], '\t', 'I-' + tag[m], '\t', 'I-' + truth[j][3], '\t', '\n'])
                            else:
                                if n == 0:
                                    lines.append(word[m][n] + '\t' + 'B-' + tag[m] + '\t' + 'O' + '\t' + '\n')
                                    # f.writelines([word[m][n], '\t', 'B-' + tag[m], '\t', '0', '\t', '\n'])
                                else:
                                    lines.append(word[m][n] + '\t' + 'I-' + tag[m] + '\t' + 'O' + '\t' + '\n')
                                    # f.writelines([word[m][n], '\t', 'I-' + tag[m], '\t', '0', '\t', '\n'])
                else:
                    for m in range(len(word)):
                        for n in range(len(word[m])):
                            num = num + 1
                            if int(truth[j][1])==num:
                                if n == 0:
                                    lines[num-1]= word[m][n] + '\t' + 'B-' + tag[m] + '\t' + 'B-' + truth[j][3] + '\t' + '\n'
                                else:
                                    lines[num-1] = word[m][n] + '\t' + 'I-' + tag[m] + '\t' + 'B-' + truth[j][3] + '\t' + '\n'
                            elif (int(truth[j][2])>= num) and (int(truth[j][1])<num):
                                if n == 0:
                                    lines[num-1]= word[m][n] + '\t' + 'B-' + tag[m] + '\t' + 'I-' + truth[j][3] + '\t' + '\n'
                                    print('BI')
                                else:
                                    lines[num-1] = word[m][n] + '\t' + 'I-' + tag[m] + '\t' + 'I-' + truth[j][3] + '\t' + '\n'
    f.writelines(lines)
    f.writelines('\n')


#释放模型
postagger.release()  # 释放模型
segmentor.release()  # 释放模型