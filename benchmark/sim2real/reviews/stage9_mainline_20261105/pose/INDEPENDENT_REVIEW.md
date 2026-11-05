# 正式评估器独立复核

复核对象v2：`de843e737562b3aeb08999006efbd955fafda039336f406b08ec61329f02667f`。v1继续保留，不覆盖正在运行的CPU接口准备。

## 已核对

- 实际Stage9训练器使用同一路径的SingleShotPose、`cfg/yolo-pose.cfg`、416×416输入和20通道输出。评估器解码直接对应训练验证的channel18全图argmax，中心sigmoid/角点线性偏移和网格归一一致。
- 使用实际训练`encode_event_window`函数做独立数值对照：新ON/OFF/total计数、log1p/固定8缩放、各通道独立线性resize和float32连续CHW逐位一致。未按每个样本/生成器最大事件数归一。
- 去畸变LUT与逐事件直接调用OpenCV的坐标结果逐位一致，使用公开标定，不使用GT区域。
- 固定8个公开CAD角点的EPnP对已知6DoF姿态恢复，旋转矩阵最大误差1.28e-15、平移1.01e-16米。GT pose、GT UV只在解码后计算误差，没有进入网络、confidence选点或PnP输入。
- prepared文件逐MAT记录原始HDF5精确索引，校验全部packet的原始xy/极性/相对时间一致；全序列名单按原始文件身份决定。没有事件稀疏度或误差驱动筛选。
- 公开CAD顶点的米制直径、9点顺序与Stage9物理bank进行准备门禁。目标scope明确限于YCB036木块；未声称整个E-POSE。
- ADD-S主方向GT→estimated经独立变换点云KD树对照与非对称有限点集验证。0.1m和0.1D两种AUC均固定全帧分母，非有限预测/PnP失败贡献0。成功率阈值严格小于0.1D。
- 评分要求完整15组30000更新收据、每seed同初始化、合成选出的权重hash、protocol hash、评估源码hash、Darknet源码/cfg清单一致；默认授权关闭。

## 已修正

v1辅助角点直接EPE是所有有限UV均值，却标记为`mean_valid_pnp`。获父会话许可后新建v2，将名称改为`corner_epe_mean_valid_uv_px`。逐帧数值与主指标完全不变。投影误差仍是PnP有效帧上的9点投影均值，单独准确命名。

## 仍须父会话核验的真实条件

本地人造数据/数值验证不替代服务器真实接口预检。需要完整确认服务器上double-block目录、原始`events_parallel[4,N]`、MAT实际字段、所有真实packet的对应关系及CAD。门禁失败须先分析，不能默默跳帧或放宽。

此外，当前授权冻结的是评分时模型源码/cfg。父会话应将这些hash与实际Stage9训练时保存的来源清单比较；strict state_dict加载可验证参数结构，无法独立证明训练/评估forward源码完全相同。

协议沿用PoseCNN的固定0.1m AUC范围，积分实现明确为经验CDF的精确解析面积，不冒称逐位复制某个MATLAB脚本或官方E-POSE榜单。旧pilot使用反向最近邻、旧训练和已暴露r05片段，不能并入本次正式结果。

评分时建议沿用父会话的单GPU和CPU资源限制；完整CAD最近邻评估可能明显消耗CPU，不能为提速临时抽点/抽帧。若运行耗时超过预期，可分开固定预测与CPU指标计算，保持同一冻结协议和完整分母。
