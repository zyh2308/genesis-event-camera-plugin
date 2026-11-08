# 位姿数据合同 CPU 审计 v2：保留失败

v2 已在服务器完成两个生成器的全部96条训练轨迹、各1529个窗口及独立验证输入统计计算。最终 json.dump 因 input_information 中 np.int64 无法序列化而退出1，旧 partial JSON 和 stdout 按原样保留；不能称本轮无失败，也不能把未完整导出的饱和统计伪装成v2完整结果。root 对 partial 中已经完整写出的 core-contract 字段另行回收，须明确该范围。

v2 锁定源码及 manifest 原字节归档于 sources，manifest SHA256 5562c888adb127af5afb363345d355e913dd8289ef917ad8a2ebd618bf8b5aa6。该失败属于报告序列化，不是训练或模型修复。v2 源和失败文件均未被v3覆盖。
