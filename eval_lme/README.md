本文件结构和eval类似，切入主文件是main_lme_parse.py

运行lme_eval.py 获得每个问题metrics

运行lme_metrics.py 获得excel和综合得分结果

demo 里面的只跑了llm judged，results里面的llm judged结果不对。要相互补充一下。
final correct result = demo内result + results内result去掉llm judged结果部分