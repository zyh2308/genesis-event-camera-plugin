# V2E采样与质量指标：原始文献及源码核查

2026-11-05。只读研究；固定代码版本为`cac638444557a224ba6ee5c29260916f152c00bd`。

**没有在最终CVPRW论文中找到与本轮完全相同的“采样步数—事件时间误差”数值表。需要自行实测，不能用真实传感器延迟或下游识别分数代替。**

## 正式论文

- §4、Fig.5，PDF第4–5页：自动插帧结合用户时间步长与最大光流，每帧运动约不超过1像素。亮度相关一阶低通、lin-log映射及噪声规则在同节说明；没有规定377步或20微秒合格线。[CVPRW 2021原文](https://tub-rip.github.io/eventvision2021/papers/2021CVPRW_V2E_From_Video_Frames_to_Realistic_DVS_Events.pdf#page=4)
- §3.3、Fig.4，PDF第3页：是引用的真实DVS对LED熄灭的响应延迟与抖动，不是V2E离散算法的误差。[Fig.4](https://tub-rip.github.io/eventvision2021/papers/2021CVPRW_V2E_From_Video_Frames_to_Realistic_DVS_Events.pdf#page=3)
- §6.1、Fig.6，PDF第6页：真实与合成明暗椅子示例作定性比较。Table2/3使用下游识别准确率和AP50，不能直接填入本轮采样精度表。[§6](https://tub-rip.github.io/eventvision2021/papers/2021CVPRW_V2E_From_Video_Frames_to_Realistic_DVS_Events.pdf#page=6)

## 更相关的早期预印本

作者官网明确区分较详细的v1与为CVPRW调整的版本，引用时应标明版本。[作者官网](https://sites.google.com/view/video2events/home)

- 2020年v1，§IV、Fig.7–8，PDF第8–9页：扫ON/OFF阈值以匹配真实DAVIS事件计数；Fig.8按100毫秒窗口比较ON/OFF数量。它提供计数统计思路，没有通用时间误差或F1数值。**其用真实数据校准阈值的流程不得套用于我们的隔离real-test集合。**[v1 §IV](https://arxiv.org/pdf/2006.07722v1#page=8)
- v1 §II、Fig.5，第4–5页：静止场景录1分钟，比较光照、带宽和阈值对ON/OFF噪声率的影响。可借鉴足够曝光及按亮度分区的方法，但不能用0.1秒零事件证明完整噪声等价。[v1噪声实验](https://arxiv.org/pdf/2006.07722v1#page=4)

CVF补充材料ZIP入口本轮返回403，未将其未读内容写成结论。

## 官方效率说明

锁定版本README称346×260、10倍插帧在GTX1050上约慢于实时50–200倍；另给极高插帧倍率案例。这是运行建议，硬件、输入、插帧倍率及统计口径均与本轮不同，不能直接计算我们的速度优势。[固定README第24–33行](https://github.com/SensorsINI/v2e/blob/cac638444557a224ba6ee5c29260916f152c00bd/README.md#L24-L33)

## 固定源码对测试的实际约束

- 原生输入按0–255 DN解释；低DN使用lin-log，低通强度系数为`(DN+20)/275`，不是我们的`clip(I,0,1)`。同一个300Hz参数不能保证暗处的连续响应相同。[工具函数](https://github.com/SensorsINI/v2e/blob/cac638444557a224ba6ee5c29260916f152c00bd/v2ecore/emulator_utils.py#L17-L103)
- Euler更新的`epsilon>0.3`触发警告，随后上限截为1；警告不是经过验证的精度合格线。`hdr=True`虽绕过lin-log，带宽系数仍从输入数值计算，不能直接视为已匹配物理log路径。
- 原生事件时间按全图该步最大事件数均匀分配；单交点通常落在输入帧时间点。需要保留此行为及FP32时间输出，不能替它改成根插值。[emulator时间生成](https://github.com/SensorsINI/v2e/blob/cac638444557a224ba6ee5c29260916f152c00bd/v2ecore/emulator.py#L711-L732)
- 源码旧注释提到初始化随机base相位，**实际下移语句在本固定版本已经注释掉**。此前仅依注释推断其生效是不正确的；本轮噪声脚本实测初始化`lp-base`偏移，不预设相位差。[初始化](https://github.com/SensorsINI/v2e/blob/cac638444557a224ba6ee5c29260916f152c00bd/v2ecore/emulator.py#L448-L470)
- 即使关闭时间噪声，信号事件输出仍可能调用`randperm`；不应用“noise-off时RNG必须不变”断言。

## 本轮可比口径

同N、时间窗及指标下，分别对各自声明的连续传感器规律计算离散误差；再对共同有效DN范围的场景报告原生完整管线误差。原测试I=1.6为HDR，不能直接乘255后当成V2E合法8bit域，也不能悄悄截断。使用固定曝光或共同有效子集时必须明确。

短噪声检查保留原生全部公式：光学I为0.01/0.1/0.5/1.0，一次固定映射DN=255I；32×32、每N三次独立时间流、每次0.1秒。固定空间种子使阈值及噪声率图一致，初始化后独立设置时间随机种子。记录初始化、概率dtype及极小ON概率的FP32支持边界；计数不足时不做分布等价或生成器排名。
