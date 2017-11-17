from UserCF import UserCf


class UsercfTime(UserCf):
    """
    原有的usercf算法结合时间过滤，只从一段时间内的新闻做推荐列表
    """

    def predict(self, data_range, top_user_k, top_answer_k):
        news_time_dict = {}
        for i in range(0, data_range[0]):
            if self.items[i].news_id not in news_time_dict:
                news_time_dict[self.items[i].news_id] = self.items[i].time_slot
            elif self.items[i].news_id in news_time_dict and self.items[i].time_slot < news_time_dict[
                self.items[i].news_id]:
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
                            if abs(news_time_dict[tmp_ans_list[t][0]] - self.items[i].time_slot) < 259200:
                                # 要求推荐的新闻时间必须在7天之内
                                with_time.append(tmp_ans_list[t])
                        answer.append([a[0] for a in with_time[:top_answer_k]])
                    else:
                        answer.append([a[0] for a in tmp_ans_list[:top_answer_k]])
                        # 选取前k个答案
                else:
                    answer.append(self.most_popular)
        return answer

    def all_test(self):
        with open("result_UserCFTime2.txt", "w", encoding="UTF-8") as f:
            for i in range(60,80):
                print("cf top_user_k = {} top_answer_k = {} right_ratio = {}".format(i, 5, self._test("cf", i, 5)))
                f.write(
                    "cf top_user_k = {} top_answer_k = {} right_ratio = {}\n".format(i, 5, self._test("cf", i, 5)))
            for i in range(1, 60):
                print(
                    "iif top_user_k = {} top_answer_k = {} right_ratio = {}".format(i, 5, self._test("iif", i, 5)))
                f.write(
                    ("iif top_user_k = {} top_answer_k = {} right_ratio = {}\n".format(i, 5, self._test("iif", i, 5))))


if __name__ == "__main__":
    uct = UsercfTime()
    uct.all_test()
