from collections import namedtuple
import bisect

item = namedtuple("item", ["user_id", "news_id", "time_slot"])


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
