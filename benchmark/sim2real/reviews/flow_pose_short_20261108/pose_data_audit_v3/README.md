# 位姿数据合同 CPU 审计 v3：序列化修复通过

v3 保留v2统计脚本原字节，仅由独立 wrapper 在 JSON 写出边界使用 NumPy generic.item / ndarray.tolist。完整 input_information 写出、超过2^53整数精确保留和 allow_nan=False 拒绝NaN的 CPU 测试通过；服务器 CPU3/3，原数据全量重算后 exit0/statuspassed。wrapper 将原v2脚本、失败 partial、自己的源码及 manifest SHA 写入来源，并再次核验旧 partial 未变。未复用缺失统计，也未覆盖v2失败。

root 反馈的实测结果：两生成器各1529个 normal raw→NPZ 窗口逐像素差异为0；全部96条训练轨迹的正常/半亮度/最暗 K、位姿、时间及关键点一致。模型128输入 exact1率各组为0，全图原始 count>=8率最大约1.72e-5，当前证据不支持裁剪饱和是主要原因。非零率等输入描述不能直接证明预测误差的成因。详细原始 JSON 及完整合同元数据仍由root归档后独立核验。

这证明只读审计完成并成功写出，不代表全部来源元数据已独立核验，更不代表位姿预测问题已经解决、候选已满足30+8门槛或真实E-POSE效果改善。sources 保存4个锁定文件及 manifest 原字节，manifest SHA256 2c2fd7c7eab3a4fc74e4840b06f7b7033042b21ad2e9dd2b81f8c892e794467。
