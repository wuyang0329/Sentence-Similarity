#encoding=utf-8

from Segment import *
from fileObject import FileObj
from sentenceSimilarity import SentenceSimilarity

if __name__ == '__main__':
    # 读入训练集
    file_obj = FileObj(r"testSet/trainSet.txt")
    train_sentences = file_obj.read_lines()
    # 读入测试集1
    file_obj = FileObj(r"testSet/testSet1.txt")
    test1_sentences = file_obj.read_lines()

    # 读入测试集2
    file_obj = FileObj(r"testSet/testSet2.txt")
    test2_sentences = file_obj.read_lines()

    # 分词工具，jieba分词，
    seg = Seg()

    # 训练模型
    ss = SentenceSimilarity(seg)
    ss.set_sentences(train_sentences)
    ss.TfidfModel()         # tfidf模型
    # ss.LsiModel()         # lsi模型
    # ss.LdaModel()         # lda模型

    # 测试集1`
    right_count = 0
    print("train_length:",len(train_sentences))
    for i in range(0,len(train_sentences)):
        #找到最相似的句子
        sentence = ss.similarity(test1_sentences[i])
        print("{} \t {} \t {:.3f}".format(test1_sentences[i], sentence.get_origin_sentence(), sentence.score))
        if i != sentence.id:
            print ("{} {}".format(str(i),"wrong!"))
        else:
            right_count += 1
            print ("{} {}".format(str(i),"right!"))

    print ("{}:{:.3f}".format("正确率为",float(right_count)/len(train_sentences)))


    s1 = "正如我们所知道的，在世界上每个国家的不同地区"
    s2 = "大家都知道，在世界上每一个国家的不同区域"

    s1_cut = seg.cut(s1)
    s2_cut = seg.cut(s2)
    #利用simhash计算两个句子之间的相似度
    simhash_simlirity = ss.simhash_distance(s1_cut,s2_cut)
    print("simhash simlirity:{:.3f}".format(simhash_simlirity))

    #wmd算法计算两个句子间的相似度
    wmd_simlirity = ss.wmd_distance(s1_cut,s2_cut)
    print("wmd simlirity:{:.3f}".format(wmd_simlirity))

