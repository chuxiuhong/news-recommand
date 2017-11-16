# 新闻推荐

数据集使用http://www.pkbigdata.com/common/cmpt/CCF%E5%A4%A7%E6%95%B0%E6%8D%AE%E7%AB%9E%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html

评价指标使用top5的预测准确率，打乱新闻顺序，使用前80%作为训练集，后20%作为测试集

### Time-Only
---------
只使用时间作为最近邻推荐，目前最佳结果为26.42%
### UserCF
---------
目前最佳结果为24.68%，top_user = 50，距离算法使用iif，对热门新闻做惩罚

### UserCF-Time
----------
使用usercf算法结合时间过滤。
目前最佳结果为27.25%，top_user = 60，距离算法同样使用iif，时间过滤要求推荐新闻在用户前后7天内

### LFM
暂时未完成

本项目文件中已ignore原始文件，而只是用去掉新闻正文的用户id和新闻id