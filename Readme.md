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
# 本体事故基本无法辨别，这个跟模型训练有关，目前有的硬件条件训练不出来