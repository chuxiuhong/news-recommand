import random


def data_process(readfile="CCF/train_data.txt", out="train_id.txt"):
    '''
    将源数据去除掉新闻内容，只保留用户id，新闻id和时间戳
    :param readfile:源数据路径 
    :param out: 输出路径
    :return: 
    '''
    with open(readfile, "r", encoding="UTF-8") as f:
        data_list = f.readlines()
    reformat_list = [" ".join(i.split()[0:3]) for i in data_list]
    random.shuffle(reformat_list)
    reformat_str = "\n".join(reformat_list)
    with open(out, "w", encoding="UTF-8") as f:
        f.write(reformat_str)


if __name__ == "__main__":
    data_process()
