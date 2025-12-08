##### AI is only use for learing how to write python in this project and detect potential bug
# Environment&Version
python version=3.10.18,
Anaconda version=25.5.1,
pytorch=2.9,
cuda=12.8
# 后台模块解耦方法
AsyncBus  异步并发（asynchronous concurrency）。
帧事件，推理事件放在并发池中，暂时无法保证抗压能力，
# 关于是否是二分类问题
显然不是，检测到会有检测框，没有则不会有，对否项置空，注意这个
# 数据库
暂时先用管理员跑通，后期有空再变普通成员
# 检测灵敏度问题，事故至少连续4个检测帧报事故才计入数据，否则建议丢弃
在30m/s对撞的时候，60fps的检测帧会有4帧事故.
使用事件聚合器，多帧相邻报事故才确认事故开始，然后写入
# 本体事故基本无法辨别
这个跟模型训练有关，目前有的硬件条件训练不出来
# 运行流程讲解
1. 启动后台
2. 本地llm常驻gpu
3. 等待前端post play

![img.png](img.png)
4. post play像前端返回sse_id，同时等待链接
5. 连接后分析指定视频，通过session会话传回前端
![img_1.png](img_1.png)
6. 分析完后前端可以选择自动断开sse
![img_4.png](img_4.png)
7. 分析完可以智能查询
![img_3.png](img_3.png)
# 前端请求字段讲解，
本地运行所以可以用post http://127.0.0.1:8000/api/play 来请求启动服务，

然后用返回的sse_alerts_url拼凑新的sse访问地址get http://127.0.0.1:8000/sse/alerts?sse_id=sse-main

用post http://127.0.0.1:8000/api/ask 请求智能回答

用post http://127.0.0.1:8000/api/stop 请求关闭sse
# 尚未完成天气预测和一些返回json的优化，请先留空

# 测试案例在train&test里面的test_picture和test_video
给三个短视频用于测试后续会找更多视频
# 训练数据在train&test里面的train_data



# 天气数据要引入
@InProceedings{bdd100k,
    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
用mobilenet_v3_small训练的
# 训练过程关键结果参数
[INFO] raw clear=51082, rain=5083
[INFO] downsample clear → 12000
[INFO] oversample rain → 12000
[Epoch 1/5] TrainLoss=114.663 TrainAcc=0.8517 | ValLoss=11.404 ValAcc=0.8650
[Epoch 2/5] TrainLoss=85.709 TrainAcc=0.8914 | ValLoss=10.360 ValAcc=0.8762
[Epoch 3/5] TrainLoss=70.039 TrainAcc=0.9135 | ValLoss=8.824 ValAcc=0.9033
[Epoch 4/5] TrainLoss=55.750 TrainAcc=0.9346 | ValLoss=8.649 ValAcc=0.9142
[Epoch 5/5] TrainLoss=42.907 TrainAcc=0.9518 | ValLoss=8.304 ValAcc=0.9237

