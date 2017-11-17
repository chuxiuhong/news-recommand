from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

item = namedtuple("item", ["user_id", "news_id", "time_slot"])

Item = []
with open("train_id.txt", "r", encoding="UTF-8") as f:
    for i in f.readlines():
        Item.append(item(i.split()[0], i.split()[1], int(i.split()[2])))
def news_time_plot():
    """
    统计新闻被阅读时间分布信息，验证新闻时效性是否明显
    实际效果极端到连长尾分布这个词都不够表示
    :return: 
    """
    time_stat = {}
    news_time = {}
    for im in Item:
        if im.news_id not in news_time:
            news_time[im.news_id] = set()
        news_time[im.news_id].add(im.time_slot)
    for news in news_time:
        t_time = list(news_time[news])
        t_time.sort()
        mean_time = sum(t_time) / len(t_time)
        for k in t_time:
            if k - mean_time in time_stat:
                time_stat[k-mean_time] += 1
            else:
                time_stat[k-mean_time] = 1
    x,y = [],[]
    for t in time_stat:
        x.append(t)
        y.append(time_stat[t])
    plt.plot(x,y)
    plt.show()

if __name__ == "__main__":
    news_time_plot()

