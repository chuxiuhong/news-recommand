from collections import namedtuple
from math import log
import bisect

item = namedtuple("item", ["user_id", "news_id", "time_slot"])


class UserCf(object):
    def __init__(self):
        self.type = "cf"
        # 选定测试时的类型
        self.items = []
        self.user_similarity_dict = {}
        # 相似矩阵的字典实现
        self.news_dict = {}
        self.user_dict = {}
        self.most_popular = ""
        # 记录最流行的新闻，用于冷启动
        with open("train_id.txt", "r", encoding="UTF-8") as f:
            for i in f.readlines():
                self.items.append(item(i.split()[0], i.split()[1], i.split()[2]))

    def user_similarity(self, data_range):
        for i in range(data_range[0], data_range[1]):
            if self.items[i].news_id not in self.news_dict:
                self.news_dict[self.items[i].news_id] = set()
            self.news_dict[self.items[i].news_id].add(self.items[i].user_id)
            if self.items[i].user_id not in self.user_dict:
                self.user_dict[self.items[i].user_id] = set()
            self.user_dict[self.items[i].user_id].add(self.items[i].news_id)
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
        times = 0
        for i in self.news_dict:
            if len(self.news_dict[i]) > times:
                times = len(self.news_dict[i])
                self.most_popular = i

    def predict(self, data_range, top_user_k, top_answer_k):
        answer = []
        # 记录推荐新闻
        for i in range(data_range[0], data_range[1]):
            sim_user = []
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
                    answer.append(tmp_ans_list[0][:top_answer_k])
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
        with open("result2.txt", "w", encoding="UTF-8") as f:
            for i in range(1, 20):
                for j in range(1, 6):
                    print("cf top_user_k = {} top_answer_k = {} right_ratio = {}".format(i, j, self._test("cf", i, j)))
                    f.write(
                        "cf top_user_k = {} top_answer_k = {} right_ratio = {}\n".format(i, j, self._test("cf", i, j)))
            for i in range(1,20):
                for j in range(1, 6):
                    print(
                        "iif top_user_k = {} top_answer_k = {} right_ratio = {}".format(i, j, self._test("iif", i, j)))
                    f.write(("iif top_user_k = {} top_answer_k = {} right_ratio = {}\n".format(i, j,
                                                                                               self._test("iif", i,
                                                                                                          j))))


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
                if p2 <0:
                    break
            else:
                point_value.append(p2)
                p1=p2
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
                p1=p2
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
            for i in (1, 3, 5,7,9):
                print("ans_num = {} right_ratio = {}".format(i, self.predcit((int(length * 0.8), length), i)))
                f.write("ans_num = {} right_ratio = {}\n".format(i, self.predcit((int(length * 0.8), length), i)))


if __name__ == "__main__":
    uc = UserCf()
    uc._test("iif",100,3)
    # tc = TimeCf()
    # tc._test()
