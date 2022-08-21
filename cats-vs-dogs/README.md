
理解任务：二分类问题

一、数据

1.1 数据下载

在Kaggle官网上[下载数据](https://www.kaggle.com/competitions/dogs-vs-cats/data)，类别标签：{“cat”: 0, "dog": 1}

```
dogs-vs-cats
	|
	|----train
	|		|----{cat/dog}.{num}.jpg, num: 0~12499
	|
	|----test1
	|		|----{num}.jpg, num: 1~12500
	|
	|----sampleSubmission.csv
```

1.2 数据准备

​	将训练集按4:1的比例划分为训练集和验证集

```
TrainingData
	|
	|----train
	|		|----cats
	|		|----dogs
	|
	|----validate
			|----cats
			|----dogs
```

train/cats目录下共计10,000幅图（train/dogs同）

validate/cats目录下共计10,000幅图（validate/dogs同）

1.3 数据读取



二、网络模型

​	五层的LeNet-5