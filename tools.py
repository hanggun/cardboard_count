"""一些工具"""
from collections import Counter
import json
import numpy as np
import sys
from tqdm import tqdm
# import redis
import re
import codecs
# import translators as ts
import cv2

sys.setrecursionlimit(1000000)


def save_json(data, output_path, encoding='utf-8'):
    """保存json文件"""
    with open(output_path, "w", encoding=encoding) as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def read_json(file, encoding='utf-8'):
    """读取json文件"""
    datasets = []
    with open(file, "r", encoding=encoding) as f:
        for line in f:
            datasets.append(json.loads(line))
    return datasets


def read_txt(file, encoding='utf-8'):
    """读取txt文件"""
    datasets = open(file, "r", encoding=encoding).read().splitlines()
    return datasets


def save_txt(data, file, split_symbol='\n', encoding='utf-8'):
    """保存txt文件"""
    with codecs.open(file, "w", encoding=encoding) as f:
        f.write(split_symbol.join(data))


def get_vocab(texts, maxlen):
    """给定列表形式的文本以及最大词表长度，获取词表"""
    counter = Counter()
    for line in texts:
        for word in line.strip():
            counter[word] += 1
    vocab = counter.most_common(maxlen)
    word, value = zip(*vocab)
    return word, value


def convert_vocab(vocab):
    """使用词表生成word2id和id2word"""

    # pylint: disable=unnecessary-comprehension
    word2id = {word: i for i, word in enumerate(vocab)}
    id2word = {i: word for i, word in enumerate(vocab)}
    return word2id, id2word


def pad_sequences(texts, word2id, oov_index=0, pad_index=0, maxlen=None):
    """输入文本，输出padding后的序列"""
    if not maxlen:
        maxlen = max([len(x) for x in texts])
    sequence = []
    for text in texts:
        tmp_seq = [word2id[x] if x in word2id else oov_index for x in text]
        if len(text) < max_len:
            tmp_seq += [pad_index] * (max_len - len(text))
        else:
            tmp_seq = tmp_seq[:max_len]
        sequence.append(tmp_seq)
    return np.array(sequence)


def sequence_padding(inputs, length=None, padding=0, mode="post"):
    """Numpy函数，将序列padding到同一长度"""
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        if mode == "post":
            pad_width[0] = (0, length - len(x))
        elif mode == "pre":
            pad_width[0] = (length - len(x), 0)
        else:
            raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, "constant", constant_values=padding)
        outputs.append(x)

    return np.array(outputs)


def data_split(data, fold, num_folds, mode):
    """划分训练集和验证集"""
    if mode == "train":
        D = [d for i, d in enumerate(data) if i % num_folds != fold]
    else:
        D = [d for i, d in enumerate(data) if i % num_folds == fold]

    if isinstance(data, np.ndarray):
        return np.array(D)
    else:
        return D


class WriteToHTML:
    """用于写入到html的函数"""

    # pylint: disable=no-self-use
    def __init__(self, f):
        style = """<style type='text/css'>
        html {
          font-family: Courier;
        }
        r {
          color: #ff0000;
        }
        g {
          color: #00ff00;
        }
        b {
          color: #0000ff;
        }
        y {
          color: #ff8c00
        }
        k {
          color: #000000
        }
        </style>"""

        self.red = "r"
        self.green = "g"
        self.blue = "b"
        self.yellow = "y"

        f.write("<html>")
        f.write(style)

    def write(self, f, t, str_):
        """写html格式"""
        f.write("<%(type)s>%(str)s</%(type)s>" % {"type": t, "str": str_})

    def write_to_html(self, f, texts=None):
        """写入到html
        示例:
        html = open("result.html", "w", encoding="utf-8")
        html_writer = tools.WriteToHTML(html)
        html_writer.write_to_html(html, texts, yp)
        html_writer.write_to_html(html) # 代表空行
        html_writer.write_to_html(html) #代表空行
        """

        if not texts:
            f.write("<br>")
        else:
            for item in texts:
                self.write(f, item['color'], item['text'])


def get_word2vec(pretrain_dir, output_array=True):
    """获取word2vec词向量字典, output_array决定是否输出每个单词以及他们的向量，False能够快速
    输出所有的单词"""
    word2vec = {}
    with open(pretrain_dir, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="获取word2vec"):
            if len(line.split()) == 2:
                continue
            if output_array:
                word2vec[line.split()[0]] = np.array(
                    [float(x) for x in line.split()[1:]], dtype=np.float32
                )
            else:
                word2vec[line.split()[0]] = 1
    return word2vec


def get_batch(dataset, batch_size):
    """将数据分成多个batch size"""
    batches = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        cur_batch = dataset[i : i + batch_size]
        batches.append(cur_batch)
    return batches


def text_segmentate(text, maxlen, seps="\n", strips=None):
    """将文本按照标点符号划分为若干个短句"""
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = "", []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ""
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]


def cut_sentences(para, drop_empty_line=True, strip=True, deduplicate=False):
    '''cut_sentences

    :param para: 输入文本
    :param drop_empty_line: 是否丢弃空行
    :param strip: 是否对每一句话做一次strip
    :param deduplicate: 是否对连续标点去重，帮助对连续标点结尾的句子分句
    :return: sentences: list of str
    '''
    if deduplicate:
        para = re.sub(r"([。！？\!\?])\1+", r"\1", para)

    para = re.sub('([。！？\?!])([^”’)\]）】])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{3,})([^”’)\]）】….])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…+)([^”’)\]）】….])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?!]|\.{3,}|\…+)([”’)\]）】])([^，。！？\?….])', r'\1\2\n\3', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sentences = para.split("\n")
    if strip:
        sentences = [sent.strip() for sent in sentences]
    if drop_empty_line:
        sentences = [sent for sent in sentences if len(sent.strip()) > 0]
    return sentences


max_len = 256


def text_split(text, limited=True):
    """将长句按照标点分割为多个子句。"""
    # pylint: disable=redefined-outer-name
    texts = text_segmentate(text, 1, u"\n。")
    if limited:
        texts = texts[:max_len]
    return texts


class WriteToRedis:
    def __init__(self):
        self.redis_client = redis.StrictRedis(host='10.0.102.72', password='admin8888', port='6379', db=3)

    def write(self, content):
        self.redis_client.lpush("fm_news_list", json.dumps(content, ensure_ascii=False))


def get_cn_ratio(text_in):
    """返回汉字占所有文字的比例"""
    cn_pattern = re.compile(r'[\u4e00-\u9fa5]')
    ratio = len(cn_pattern.findall(text_in)) / len(text_in)
    return ratio


def is_cn(word):
    """返回文字是否为中文"""
    cn_pattern = re.compile(r'[\u4e00-\u9fa5]+')
    if cn_pattern.match(word):
        return True
    else:
        return False


def cut_sentence(text_in, max_length):
    """将文本按照句号和换行符进行切割，切割后将标点符号与前一个句子整合起来"""
    pattern = re.compile('([。？?!！;；])')
    sentences = pattern.split(text_in)
    if len(sentences) < 2:
        return sentences
    clue_sentences = []
    # 若句子长于max_length，则按照max_length分成多个句子
    for i in range(1, len(sentences), 2):
        if len(sentences[i - 1]) < max_length:
            clue_sentences.append(sentences[i - 1] + sentences[i])
        else:
            for j in range(0, len(sentences[i - 1]), max_length):
                clue_sentences.append(sentences[i - 1][j:j + max_length])
            clue_sentences[-1] = clue_sentences[-1] + sentences[i]
    # 对最后一个落单的句子做同样的操作
    if i != len(sentences) - 1:
        if sentences[len(sentences) - 1]:
            if len(sentences[len(sentences) - 1]) < max_length:
                clue_sentences.append(sentences[len(sentences) - 1])
            else:
                for j in range(0, len(sentences[len(sentences) - 1]), max_length):
                    clue_sentences.append(sentences[len(sentences) - 1][j:j + max_length])

    return clue_sentences


def cut_sentence_with_fix_length(text_in, max_length):
    texts = []
    for i in range(0, len(text_in), max_length-2):
        texts.append(text_in[i:i+max_length-2])
    return texts


class DeDuplicate:
    def __init__(self):
        idf = tools.read_txt(business_config.idf_dir)
        self.idf = {}
        for line in idf:
            line = line.split()
            self.idf[line[0]] = float(line[1])
        self.dup_dict = {}

    def _get_pos_res(self, title):
        pos_title = pseg.lcut(title)
        cut_title = sorted(
            [
                (pair.word, self.idf[pair.word])
                for pair in pos_title
                if pair.word in self.idf
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        return pos_title, cut_title

    def __call__(self, content: str):
        """Return True if the title is duplicate else False
        The input content is a dictionary with keys of title and publish_time
        """
        dup_title = ""
        if not content:
            return {"flag": True, "title": "title is empty"}

        pos_title, cut_title = self._get_pos_res(content)
        # pylint: disable=too-many-nested-blocks
        for word, _ in cut_title:
            if word not in self.dup_dict:
                self.dup_dict[word] = {"pos_titles": [pos_title]}
                return {"flag": False, "title": dup_title}
            else:
                dup_flag = False
                for _, tmp_pos_title in enumerate(self.dup_dict[word]["pos_titles"]):
                    dup_mark_num = 0
                    diff = set(tmp_pos_title).symmetric_difference(set(pos_title))
                    for diff_element in diff:
                        if diff_element.flag in ["m", "n", "eng"]:
                            dup_mark_num = 4
                            break
                        else:
                            if diff_element.flag != "x":
                                dup_mark_num += 1
                    if dup_mark_num <= 3:
                        dup_flag = True
                        dup_title = "".join([x.word for x in tmp_pos_title])
                        break

                if not dup_flag:
                    self.dup_dict[word]["pos_titles"].append(pos_title)
                    return {"flag": False, "title": dup_title}
                else:
                    return {"flag": True, "title": dup_title}

        return {"flag": True, "title": "no word in idf table"}


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def show_picture(image):
    cv2.imshow("Image", image)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def translate(text, to_language='en'):
    return ts.translate_text(text, to_language=to_language)

if __name__ == '__main__':
    print(cut_sentence('莱芜高新区2018年道路交通安全隐患整改项目施工中标公告Ж一、采购人:莱芜高新技术产业开发区管委会Ж地Ж址:莱芜高新区汇源大街108号Ж联系人:李主任Ж联系方式:0634-5667834'
                 'Ж二、采购代理机构:山东嘉信建设管理咨询有限公司Ж三、采购项目名称:莱芜高新区2018年道路交通安全隐患整改项目施工Ж四、采购公告发布日期:2018年9月14日至2018年9月20'
                 '日Ж五、开标日期:2018年10月10日Ж六、采购方式:公开招标Ж七、中标情况:Ж预中标供应商名称Ж中标价Ж地Ж址Ж业绩公示Ж莱芜市众智电子工程有限公司Ж899880.00'
                 'Ж莱芜市莱城区董花园乐园街8号Ж1、莱芜高新区内道路交通信号灯等设施安装及施工('
                 '2016年)Ж2、企业获得质量、环境管理体系认证Ж八、公示期:2018年10月13日—2018年10月15'
                 '日Ж九、评标小组成员名单:赵辉、杨新华、宋倩、李明、魏益进。Ж十、评审结果:Ж排序Ж供应商名称Ж报价('
                 '元)Ж工期Ж评审结果Ж第一中标候选人Ж莱芜市众智电子工程有限公司Ж899880.00Ж50天Ж中标Ж第二中标候选人Ж山东省信息产业服务有限公司Ж846040.00Ж30'
                 '天Ж未中标Ж第三中标候选人Ж山东力盟电力电子有限公司Ж883820.00Ж50天Ж未中标Ж十一、联系方式Ж采购代理机构:山东嘉信建设管理咨询有限公司Ж地Ж址:莱芜市鹏泉东大街68'
                 '号嘉信大厦Ж联系人:刘先生Ж联系方式:0634-6228188Ж发Ж布Ж人:山东嘉信建设管理咨询有限公司Ж发布时间:2018年10月12日Ж['
                 '高新区]莱芜高新区2018年道路交通安全隐患整改项目施工中标公示', 128))