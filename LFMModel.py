import numpy as np
from collections import namedtuple

item = namedtuple("item", ["user_id", "news_id", "time_slot"])


class LfmModel(object):
    def __init__(self, K, epoch, alpha, lambda_, train_data_ratio, top_k, select_ratio):
        """
        LFM模型的初始化，其核心思想是找到物品的隐类型，分别算出用户与隐类型、隐类型和物品的关系
        :param K: 隐类型个数
        :param epoch: 训练轮数
        :param alpha: 训练步长
        :param lambda_: 惩罚项系数
        :param train_data_ratio:训练集占总数据比例 
        :param top_k: topk推荐的k
        :param select_ratio: 每个用户新闻负正样本比例
        """
        self.items = []  # 记录全部数据
        self.news = set()
        self.K = K
        self.alpha = alpha
        self.lambda_ = lambda_
        self.epoch = epoch
        self.train_data_ratio = train_data_ratio
        self.top_k = top_k
        self.select_ratio = select_ratio
        self.user_dict = {}
        self.all_users = []
        if type(select_ratio) != int or select_ratio < 1:
            raise ValueError("select_ratio must be int and over than 1")
        with open("train_id.txt", "r", encoding="UTF-8") as f:
            for i in f.readlines():
                self.items.append(item(i.split()[0], i.split()[1], i.split()[2]))
        tmp_list = []
        for i in range(int(len(self.items) * self.train_data_ratio)):
            if self.items[i].user_id not in self.user_dict:
                self.user_dict[self.items[i].user_id] = set()
            self.user_dict[self.items[i].user_id].add(self.items[i].news_id)
            if self.items[i].news_id not in self.news:
                tmp_list.append((self.items[i].news_id, int(self.items[i].time_slot)))
                self.news.add(self.items[i].news_id)
        self.news = tmp_list
        self.news.sort(key=lambda x: x[0])
        self.news_sort_time = self.news[:]
        self.news_sort_time.sort(key=lambda x: x[1])
        self.p = np.random.random((len(self.user_dict), K))  # 用户与隐类型的兴趣
        self.q = np.random.random((len(self.news), K))  # 隐类型与新闻的关系

    def _find_news(self, news_id):
        l = 0
        r = len(self.news) - 1
        while self.news[l][0] <= self.news[r][0]:
            mid = (l + r) // 2
            if self.news[mid][0] == news_id:
                return mid
            elif self.news[mid][0] > news_id:
                r = mid - 1
            else:
                l = mid + 1

    def _find_user(self, user_id):
        l = 0
        r = len(self.all_users) - 1
        while self.all_users[l] <= self.all_users[r]:
            mid = (l + r) // 2
            if self.all_users[mid] == user_id:
                return mid
            elif self.all_users[mid] > user_id:
                r = mid - 1
            else:
                l = mid + 1

    def _search_negative(self, user_id, news_id):
        """
        按比例获取负样本
        :param user_id:用户id 
        :param news_id: 用户阅读的新闻id
        :return: 新闻的位置，负样本列表
        """
        mid = self._find_news(news_id)
        n = 0
        ans = []
        l = mid - 1
        r = mid + 1
        while n < self.select_ratio:
            if l >= 0:
                if self.news[l] not in self.user_dict[user_id]:
                    ans.append(l)
                    n += 1
                    if n == self.select_ratio:
                        break
            if r < len(self.news):
                if self.news[r] not in self.user_dict[user_id]:
                    ans.append(r)
                    n += 1
            l -= 1
            r += 1
        return mid, ans

    def get_sample(self, user_id):
        sample_list = []
        rating = []
        for p_news in self.user_dict[user_id]:
            mid, ans = self._search_negative(user_id, p_news)
            sample_list.append(mid)
            sample_list.extend(ans)
            rating.append(1.0)
            rating.extend([0.0] * self.select_ratio)
        return sample_list, rating

    def get_candidate(self, user_id):
        candidate = []
        for news_id in self.user_dict[user_id]:
            mid = self._find_news(news_id)
            n = 0
            l = mid - 1
            r = mid + 1
            while n < 10:
                if l >= 0:
                    if self.news[l] not in self.user_dict[user_id]:
                        candidate.append(self.news[l][0])
                        n += 1
                        if n == self.select_ratio:
                            break
                if r < len(self.news):
                    if self.news[r] not in self.user_dict[user_id]:
                        candidate.append(self.news[r][0])
                        n += 1
                l -= 1
                r += 1
        return candidate

    def train(self):
        self.all_users = list(self.user_dict)
        self.all_users.sort()
        for step in range(self.epoch):
            for i in range(len(self.all_users)):
                sample, rating = self.get_sample(self.all_users[i])
                for j in range(len(sample)):
                    err = rating[j] - np.dot(self.p[i], self.q[sample[j]])
                    self.p[i] += self.alpha * (err * self.q[sample[j]] - self.lambda_ * self.p[i])
                    # print("delta" + str(self.alpha * (err * self.q[sample[j]] - self.lambda_ * self.p[i])))
                    self.q[sample[j]] += self.alpha * (err * self.p[i] - self.lambda_ * self.q[sample[j]])
                    # print(self.p)
            self.alpha *= 0.9
        print("train model complete!")
        print(self.p)
        print(self.q)

    def predict(self):
        # data_num = len(self.items) - int(len(self.items) * self.train_data_ratio)
        right = 0
        data_num = 2000
        for i in range(int(len(self.items) * self.train_data_ratio),
                       int(len(self.items) * self.train_data_ratio) + 2000):
            user = self._find_user(self.items[i].user_id)
            if user is None:
                data_num -= 1
                continue
            news_rating = []
            candidate = self.get_candidate(self.items[i].user_id)
            for j in range(len(candidate)):
                news_rating.append((candidate[j], np.dot(self.p[user], self.q[self._find_news(candidate[j])])))
            news_rating.sort(key=lambda x: x[1], reverse=True)
            for nr in news_rating[:self.top_k]:
                if nr[0] == self.items[i].news_id:
                    right += 1
                    break
        return right / data_num


def lfm_test():
    lm = LfmModel(K=100, epoch=20, alpha=0.02, lambda_=0.01, train_data_ratio=0.8, top_k=5, select_ratio=2)
    lm.train()
    print(lm.predict())
