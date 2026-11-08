# k12 注意力尾部：事后数学分解

只读取已通过独立核验的576张预定k12 map、去重native mask及原full对应记录；不读取新真实分数，不修改任何冻结包，不执行模型前向/训练。输入 verifier completion SHA、其raw-map/mask/index SHA和full completion SHA均绑定，结束再核原字节。

使用原16×16 cell centre grid，native像素尺度为346×260。区域为endpoint与endpoint_union_approx的fractional空间占比；不是特征感受野，也不是每事件真实来源。映射独立重算原cvLINEAR416→CPU AREA128→AREA16。为概率代数，将原FP32 map按其FP64求和归一化，记录原sum；只用于数学恒等式，并核map重构UV与原full UV误差≤原2e-6，不改模型预测。

`UV = mass_fg UV_fg + mass_bg UV_bg`；`e_total = (UV_fg−GT) + mass_bg(UV_bg−UV_fg)`。报告两项沿总error方向的有符号投影，负数原样保留为抵消。原始区域或总error为零导致未定义的项保留null和可用数量，不用0冒充测量。向量及投影恒等式误差≤1e-10 native px。GT使用full中既存physical endpoint anchor，不冒称与旧normalized-FP32标签逐位相同，更不称为GT修正后的定位成绩。

输出576条明细（各含两区域）、192条按轨迹汇总的三seed小表；峰在区域内且尾部正投影仅是当前注意力分布的代数性质，不能证明背景事件物理成因、训练根因或新方案效果。不选最优cell，不引入训练门槛。

三项纯CPU代数fixture已在本地WSL实际通过；不读项目数据，无模型前向或GPU。

```text
python3 -B -m unittest -v test_decompose_tail
python3 -B decompose_tail.py --verifier-completion <verifier/completion.json> --verifier-sha256 <SHA> --full-completion <full/completion.json> --full-sha256 <SHA> --out <fresh_output>
```
