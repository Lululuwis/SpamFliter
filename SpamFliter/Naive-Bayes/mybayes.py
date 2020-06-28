import numpy as np
import re
import random

"""
函数说明：利用贝叶斯分类
参数说明：
	vec2Classify - 待分类的词条数组
	p0Vec - 正常邮件类的条件概率数组
	p1Vec - 垃圾邮件类的条件概率数组
	pClass1 - 文档属于垃圾邮件的概率
返回：
	0 - 属于正常邮件类
	1 - 属于垃圾邮件类
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # p1 = reduce(lambda x, y: x * y, vec2Classify * p1Vec) * pClass1  # 对应元素相乘
    # p0 = reduce(lambda x, y: x * y, vec2Classify * p0Vec) * (1.0 - pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


"""
函数说明：利用贝叶斯训练函数
参数说明：
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
返回：
    p0Vect - 正常邮件类的条件概率数组
    p1Vect - 垃圾邮件类的条件概率数组
    pAbusive - 文档属于垃圾邮件类的概率
"""
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMat)  # 训练的文档数目
    numWords = len(trainMatrix[0])  # 每片文档的词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于垃圾邮件类的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)  # 创建numpy.ones数组，词条出现数初始化为1，使用拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0  # 使用拉普拉斯平滑，分母初始化为2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 统计属于垃圾邮件的条件概率所需数据
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 统计属于正常邮件的条件概率所需数据
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)  # 取对数，防止下溢
    return p0Vect, p1Vect, pAbusive  # 返回属于正常邮件的条件概率数组，属于垃圾邮件的条件概率数组，文档数据垃圾邮件的概率


"""
函数说明：根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
参数说明：
    vocabList-createVocabList返回的列表
    inputSet-切分的词条列表
返回：
    returnVec-文档向量，词集模型
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个0向量
    for word in inputSet:  # 遍历词条
        if word in vocabList:  # 如果词条存在于词汇表中，则置为1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in the Vocabulary!" % word)
    return returnVec  # 返回文档向量


"""
函数说明：将切分的实验样本词条整理成不重复的词汇表，即去除重复的词汇
返回：不重复的词汇表
"""
def createVocabList(dataset):
    vocabset = set([])  # 创建一个空的不重复的列表集合
    for document in dataset:
        vocabset = vocabset | set(document)
    return list(vocabset)


"""
函数说明：接受一个大字符串，并将其解析为字符串列表
返回：该字符串列表
"""
def textParse(bigString):
    listOfTokens = re.split(r"\W+", bigString)  # 将特殊符号作为切分标志进行字符串切分
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 将单词变为小写，单个字母除外


if __name__ == '__main__':
    docList = []  # 文件列表
    classList = []  # 对doclist中的文件进行垃圾邮件（1）和正常邮件（0）进行标记
    fullText = []
    for i in range(1, 26):  # 遍历ham和spam中的25个txt文件
        wordList = textParse(open('email/spam/%d.txt' % i, 'r', errors='ignore').read())  # 读取每个垃圾邮件，并将字符串转换为字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, 'r', errors='ignore').read())  # 读取每个正常邮件，并将字符串转换为字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # 创建不重复的词汇表
    trainingSet = list(range(50))
    testSet = []  # 创建存储训练集和测试集的索引列表
    for i in range(15):  # 此处在50个邮件中随机选40个做为训练集，10个作为测试集
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索引值
        testSet.append(trainingSet[randIndex])  # 添加测试集的索引值
        del (trainingSet[randIndex])  # 在训练集列表中删除并添加到测试集的索引值
    trainMat = []
    trainClasses = []  # 创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  # 将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])  # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  # 训练贝叶斯模型
    errorCount = 0  # 错误分类计数
    for docIndex in testSet:  # 遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])  # 测试集的词集模型
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加一
    print("accurrcy:%.2f%%" % (100 - (float(errorCount) / len(testSet) * 100)))
