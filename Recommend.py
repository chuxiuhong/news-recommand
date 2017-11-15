import bisect
from collections import namedtuple
from math import log

import numpy as np

item = namedtuple("item", ["user_id", "news_id", "time_slot"])


class UserCf(object):
    def __init__(self):
        self.type = "cf"
        # 选定测试时的类型
        self.items = []
        # 初始化新闻列表，元素是item那个nametuple形式
        self.user_similarity_dict = {}
        # 相似矩阵的字典实现
        self.news_dict = {}
        # 记录新闻与用户的倒排字典
        self.user_dict = {}
        # 记录用户与新闻的倒排字典
        self.most_popular = ""
        # 记录最流行的新闻，用于冷启动
        with open("train_id.txt", "r", encoding="UTF-8") as f:
            for i in f.readlines():
                self.items.append(item(i.split()[0], i.split()[1], int(i.split()[2])))

    def user_similarity(self, data_range):
        for i in range(data_range[0], data_range[1]):
            if self.items[i].news_id not in self.news_dict:
                self.news_dict[self.items[i].news_id] = set()
            self.news_dict[self.items[i].news_id].add(self.items[i].user_id)
            if self.items[i].user_id not in self.user_dict:
                self.user_dict[self.items[i].user_id] = set()
            self.user_dict[self.items[i].user_id].add(self.items[i].news_id)
            # 将两个字典初始化
        if self.type == "cf":
            # 常规的usercf算法
            for i in self.user_dict:
                for j in self.user_dict[i]:
                    for k in self.news_dict[j]:
                        if k != i:
                            if i not in self.user_similarity_dict:
                                self.user_similarity_dict[i] = {}
                            if k not in self.user_similarity_dict[i]:
                                self.user_similarity_dict[i][k] = 1
                            else:
                                self.user_similarity_dict[i][k] += 1
            for i in self.user_similarity_dict:
                for j in self.user_similarity_dict[i]:
                    self.user_similarity_dict[i][j] /= len(self.user_dict[i]) * len(self.user_dict[j])
                    # usercf的评分，用的是普通的余弦相似度
        elif self.type == "iif":
            # 增加了对热门新闻惩罚项的usercf算法
            for i in self.user_dict:
                for j in self.user_dict[i]:
                    for k in self.news_dict[j]:
                        if k != i:
                            if i not in self.user_similarity_dict:
                                self.user_similarity_dict[i] = {}
                            if k not in self.user_similarity_dict[i]:
                                self.user_similarity_dict[i][k] = 1 / log(1 + len(self.news_dict[j]))
                            else:
                                self.user_similarity_dict[i][k] += 1 / log(1 + len(self.news_dict[j]))
                                # 在usercf上增加了热门惩罚
        times = 0
        for i in self.news_dict:
            if len(self.news_dict[i]) > times:
                times = len(self.news_dict[i])
                self.most_popular = i
                # 算出最大点击量的新闻，用于冷启动

    def predict(self, data_range, top_user_k, top_answer_k):
        answer = []
        # 记录推荐新闻
        for i in range(data_range[0], data_range[1]):
            sim_user = []
            # 选取userk个相似用户
            tmp_ans_dict = {}
            if self.items[i].user_id not in self.user_similarity_dict:
                answer.append(self.most_popular)
            else:
                for j in self.user_similarity_dict[self.items[i].user_id]:
                    sim_user.append((j, self.user_similarity_dict[self.items[i].user_id][j]))
                if len(sim_user) > top_user_k:
                    sim_user = sim_user[:top_user_k]
                for u in sim_user:
                    for k in self.user_dict[u[0]]:
                        if k not in self.user_dict[self.items[i].user_id]:
                            if k not in tmp_ans_dict:
                                tmp_ans_dict[k] = u[1]
                            else:
                                tmp_ans_dict[k] += u[1]
                if len(tmp_ans_dict) > 1:
                    tmp_ans_list = []
                    for m in tmp_ans_dict:
                        tmp_ans_list.append((m, tmp_ans_dict[m]))
                    tmp_ans_list.sort(key=lambda x: x[1], reverse=True)
                    # 以评分高低排序
                    answer.append([a[0] for a in tmp_ans_list[:top_answer_k]])
                    # 选取前k个答案
                else:
                    answer.append(self.most_popular)
        return answer

    def _test(self, sim_type, top_user_k, top_answer_k):
        self.type = sim_type
        length = len(self.items)
        train_range = (0, int(length * 0.8))
        test_range = (int(length * 0.8), length)
        self.user_similarity(train_range)
        answer = self.predict(test_range, top_answer_k=top_answer_k, top_user_k=top_user_k)
        answer_num = (test_range[1] - test_range[0])  # * top_answer_k
        right = 0
        for i in range(test_range[0], test_range[1]):
            if self.items[i].news_id in self.news_dict:
                if self.items[i].news_id in answer[i - test_range[0]]:
                    right += 1
            else:
                answer_num -= 1
        # print("right ratio = " + str(right / answer_num))
        return right / answer_num

    def all_test(self):
        with open("result.txt", "w", encoding="UTF-8") as f:
            for i in range(1, 20):
                print("cf top_user_k = {} top_answer_k = {} right_ratio = {}".format(i, 5, self._test("cf", i, 5)))
                f.write(
                    "cf top_user_k = {} top_answer_k = {} right_ratio = {}\n".format(i, 5, self._test("cf", i, 5)))
            for i in range(1, 20):
                print(
                    "iif top_user_k = {} top_answer_k = {} right_ratio = {}".format(i, 5, self._test("iif", i, 5)))
                f.write(
                    ("iif top_user_k = {} top_answer_k = {} right_ratio = {}\n".format(i, 5, self._test("iif", i, 5))))
class UsercfTime(UserCf):
    def predict(self, data_range, top_user_k, top_answer_k):
        news_time_dict = {}
        for i in range(0,data_range[0]):
            if self.items[i].news_id not in news_time_dict:
                news_time_dict[self.items[i].news_id] = self.items[i].time_slot
            elif self.items[i].news_id in news_time_dict and self.items[i].time_slot < news_time_dict[self.items[i].news_id]:
                news_time_dict[self.items[i].news_id] = self.items[i].time_slot
        answer = []
        # 记录推荐新闻
        for i in range(data_range[0], data_range[1]):
            sim_user = []
            # 选取userk个相似用户
            tmp_ans_dict = {}
            if self.items[i].user_id not in self.user_similarity_dict:
                answer.append(self.most_popular)
            else:
                for j in self.user_similarity_dict[self.items[i].user_id]:
                    sim_user.append((j, self.user_similarity_dict[self.items[i].user_id][j]))
                if len(sim_user) > top_user_k:
                    sim_user = sim_user[:top_user_k]
                for u in sim_user:
                    for k in self.user_dict[u[0]]:
                        if k not in self.user_dict[self.items[i].user_id]:
                            if k not in tmp_ans_dict:
                                tmp_ans_dict[k] = u[1]
                            else:
                                tmp_ans_dict[k] += u[1]
                if len(tmp_ans_dict) > 1:
                    tmp_ans_list = []
                    for m in tmp_ans_dict:
                        tmp_ans_list.append((m, tmp_ans_dict[m]))
                    tmp_ans_list.sort(key=lambda x: x[1], reverse=True)
                    # 以评分高低排序
                    if len(tmp_ans_list) > 2 * top_answer_k:
                        with_time = []
                        for t in range(len(tmp_ans_list)):
                            if news_time_dict[tmp_ans_list[t]]
                    answer.append([a[0] for a in tmp_ans_list[:top_answer_k]])
                    # 选取前k个答案
                else:
                    answer.append(self.most_popular)
        return answer

class TimeCf(object):
    def __init__(self):
        self.items = []
        self.sort_items = []
        self.news_dict = {}
        self.time_list = []
        with open("train_id.txt", "r", encoding="UTF-8") as f:
            for i in f.readlines():
                self.items.append(item(int(i.split()[0]), int(i.split()[1]), int(i.split()[2])))

    def train(self, data_range):
        self.sort_items = self.items[data_range[0]:data_range[1]]
        self.sort_items.sort(key=lambda x: x.time_slot)
        for i in range(data_range[0], data_range[1]):
            if self.items[i].news_id not in self.news_dict:
                self.news_dict[self.items[i].news_id] = set()
            self.news_dict[self.items[i].news_id].add(self.items[i].user_id)
        for i in self.sort_items:
            self.time_list.append(i.time_slot)

    def bi_search(self, time, ans_num):
        point = bisect.bisect_left(self.time_list, time)
        length = len(self.time_list)
        point_value = []
        tmp = 1
        p1 = point
        p2 = point
        while tmp < ans_num:
            if self.sort_items[p1].news_id == self.sort_items[p2].news_id:
                p2 -= 1
                if p2 < 0:
                    break
            else:
                point_value.append(p2)
                p1 = p2
                tmp += 1
        tmp = 1
        p1 = point
        p2 = point
        while tmp < ans_num:
            if self.sort_items[p1].news_id == self.sort_items[p2].news_id:
                p2 += 1
                if p2 >= length:
                    break
            else:
                point_value.append(p2)
                p1 = p2
                tmp += 1
        return tuple(point_value)

    def predcit(self, data_range, ans_num):
        answer_num = data_range[1] - data_range[0]
        right = 0
        for i in range(data_range[0], data_range[1]):
            if self.items[i].news_id not in self.news_dict:
                answer_num -= 1
            else:
                point_tuple = self.bi_search(self.items[i].time_slot, ans_num)
                for x in point_tuple:
                    if self.sort_items[x].news_id == self.items[i].news_id:
                        right += 1
                        break
        return right / answer_num

    def _test(self):
        length = len(self.items)
        self.train((0, int(length * 0.8)))
        with open("result_time.txt", "w", encoding="UTF-8") as f:
            for i in (1, 3, 5, 7, 9):
                print("ans_num = {} right_ratio = {}".format(i, self.predcit((int(length * 0.8), length), i)))
                f.write("ans_num = {} right_ratio = {}\n".format(i, self.predcit((int(length * 0.8), length), i)))


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


if __name__ == "__main__":
    # uc = UserCf()
    # uc.all_test()
    # tc = TimeCf()
    # tc._test()
    lfm_test()
