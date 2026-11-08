# 全窗口错峰诊断，CPU、零新推理

只复用 RC/full 已完成对照中的原始 C1 attention、原始 control UV 和物理标签；忽略已停止 softfocus 的全部候选结果。SUP/full_v2 的原始 C1/center/raw 9216行中筛出1152行；与RC的两个生成器×三seed×三照度×四轨迹×16窗逐ID配对。RC/SUP模型状态记录、全部12个物理H5的来源SHA必须一致；所有所用completion、数据artifact、区域源码和H5在诊断始末核SHA。

仅在CPU从原H5重新构造全部64窗口的可见区域，动态识别目标actor、核21帧计数/物理/时间；不硬编码对象ID。复用冻结event_support.py SHA6102a77…的原416→128→16区域fractional映射，区分：峰native点在目标内；native点在外但峰cell有目标支持；endpoint cell无支持而端点union cell有支持；两者cell均无目标支持。最后一种仍不是逐事件背景来源证明，union只是时间近似。绝不只由一次native点在mask外就宣称网络错认背景。

报告全1152个预测实例的峰/原中心→CAD anchor误差、16/128硬格最近点参考、可见几何、边界、运动、事件区域支持与同模型跨照度特征cos。硬格参考只约束离散峰，不能当连续softargmax误差下界；不调参、不改decoder、不用GT改模型输入。平坦/错峰/零事件窗口保留。TRAIN没有完整峰图就不推断TRAIN峰机制；已知TRAIN/VAL误差差距须保留其原采样范围。

输出72个seed×rid、24个三seed-rid及6个条件完整表。16窗口/轨迹、四轨迹和三seed均完整，因此均值可按该固定平衡库存等权；不做p值、不把相邻窗口和模型seed当独立事件样本。事件支持另按method/light/physical-window去seed重数，384个唯一记录及6组64窗的事件加权计数/保留率，不把事件数乘3。

已有PG四格GT中心/深度因子表继续复用，不在本脚本重复CAD评分或模型forward；它是固定R的GT诊断，不涵盖中心→crop→R的全部中介作用。当前脚本不创建新候选、不启动训练、不改变原30+8或真实隔离。

```text
PY -B -m unittest -v test_diagnose_peaks
PY -B diagnose_peaks.py --readout-root RC/full --readout-sha 77a06dbedd0c8c63fe46f4c39d7c4c83bf53681931cae1be1f3cee6da053c154 --support-root SUP/full_v2 --support-sha 16be2e8e12c23b4791bdf312be1268621d56e99ff4893f0bff6e6eb2876e9e6c --out NEW
```

CPU上限600秒；失败保存新failure并退出非0，不重试或修改冻结输入。仅需NumPy；冻结区域helper另使用已有OpenCV/Torch CPU算子，CUDA在本进程隐藏，不初始化模型或GPU。三个CPU fixture覆盖native点/cell/时间union区分及错配拒绝；完整服务器数据尚待根会话实际运行。
