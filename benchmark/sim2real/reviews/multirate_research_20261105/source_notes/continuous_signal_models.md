# 连续亮度与闭式传感器的新近工作

核查日期：2026-11-05至2026-11-06。以下为原文与官方仓库阅读，未执行源码或实验。

## TIDES

原文：[arXiv2606.02058v1](https://arxiv.org/html/2606.02058v1)；[作者出版目录](https://personalpages.surrey.ac.uk/s.hadfield/papers.php)列为ECCV2026；[项目](https://tides-event-sim.github.io/)。

使用动态4DGS，输出亮度、时间导数及可见性相关量，按风险安排时间查询；局部模型内预测多个阈值穿越。其tile arbiter描述读出吞吐/jitter/drop，与我们300Hz光感受器低通不是同一个带宽。

[官方仓库](https://github.com/ThirgoodC/TIDES/tree/9e6e38b5b6886d6cd3e6088f44e7e90255c10e6d)固定SHA为9e6e38b5b6886d6cd3e6088f44e7e90255c10e6d。Git tree只有README，内容为项目名，不能当已发布实现。学习导数与可见性分段思想，不在本支线把Genesis渲染器换成学习型场景表示，也不照搬coverage suppression规则。

## FracEvent

原文：[v4，2026-10-21](https://arxiv.org/html/2606.26636v4)，页面标注WACV2027接收。闭式状态采用区间固定tau、线性log输入，再对活跃像素做二分求根。分数阶记忆是新的传感器模型，论文明确不含完整光子/泄漏随机性、阈值失配、不应期及读出。

[官方代码](https://github.com/bonaparte233/FracEvent/tree/1fe6ee7e8e7b4cbc3d2526aa80c033570bb2637f)固定于2026-08-25提交；此代码早于论文v4，不能把两者当同一个冻结版本。

源码入口：
- [core/dynamics.py](https://github.com/bonaparte233/FracEvent/blob/1fe6ee7e8e7b4cbc3d2526aa80c033570bb2637f/core/dynamics.py)：固定tau的线性输入闭式推进。
- [core/event_generation.py](https://github.com/bonaparte233/FracEvent/blob/1fe6ee7e8e7b4cbc3d2526aa80c033570bb2637f/core/event_generation.py)：先按区间终值决定ON/OFF数量，再定位各阈值时间；只有事件数量超cap才递归分半。到最小dt仍超cap会clamp。论文附录说明单调子区间条件；代码没有通用内部极值检测，因此不能保证找出所有“先ON后OFF、终值回原处”的事件。
- [core/config.py](https://github.com/bonaparte233/FracEvent/blob/1fe6ee7e8e7b4cbc3d2526aa80c033570bb2637f/core/config.py)：默认FP32、12次二分、64事件cap、noise关闭。不能继承这些数值/上限设置通过我们的严格门禁。

可迁移的是给定输入表示后的状态公式与活跃像素求根。必须另处理亮度依赖tau、单调分段、完整事件数及噪声/不应期。

## EventSVGF

[作者原文页](https://cs.dartmouth.edu/~wjarosz/publications/kim26difference-aware.html)与[官方FalcorComp](https://github.com/juhyeonkim95/FalcorComp/tree/b0f2ce3a9d835d95d66b430a555bba1e6b95b6f1)，master固定SHA为b0f2ce3a9d835d95d66b430a555bba1e6b95b6f1。

研究对象是蒙特卡洛渲染的亮度差噪声。2spp表示每像素空间光线路径预算，与377个传感器时间步无直接对应。[官方EventGenerator合同](https://falcorcomp.readthedocs.io/en/latest/src/plugin_reference/event/EventGenerator.html)输出每帧有符号事件计数，没有帧内时间戳/顺序、不应期、光感受器低通、shot/leak等。当前RTX渲染架构也不能直接套到V100S。只把相关采样/差分去噪作为将来原生路径追踪方向的线索。

## FastESIM

[大学正式条目](https://repository.tudelft.nl/record/uuid:2f3427d8-7779-41ab-8915-ca38e58edb1f)，R.J.N. Zwikker，2026年硕士论文。摘要提出用低频帧加深度/flow建立三角网格连续像素响应，直接生成事件，避免显式中间帧。

本轮核到官方摘要；18.6MB全文未完成读取，未找到可核的官方实现提交。因此不引用其完整算法/噪声保证或实时倍数作为选择依据。该条目提供了与我们“连续亮度表示”相近的研究方向，证据成熟度低于已审计源码。
