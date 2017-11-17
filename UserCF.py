from collections import namedtuple
from math import log

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
            for i in range(50,60):
                print("cf top_user_k = {} top_answer_k = {} right_ratio = {}".format(i, 5, self._test("cf", i, 5)))
                f.write(
                    "cf top_user_k = {} top_answer_k = {} right_ratio = {}\n".format(i, 5, self._test("cf", i, 5)))
            for i in range(50, 60):
                print(
                    "iif top_user_k = {} top_answer_k = {} right_ratio = {}".format(i, 5, self._test("iif", i, 5)))
                f.write(
                    ("iif top_user_k = {} top_answer_k = {} right_ratio = {}\n".format(i, 5, self._test("iif", i, 5))))


if __name__ == "__main__":
    uc = UserCf()
    uc.all_test()
