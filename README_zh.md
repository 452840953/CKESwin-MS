# CKESwin-MS 主体代码

这份代码对应当前 KBS v5.24 论文的主体方法：**解剖知识引导的视觉表征 → 独立质谱随机森林 → 同标本分数级融合**。

本目录只包含源码、配置模板、说明文档和合成测试代码，**不含权重、数据、标本名单、实验结果、预测表或训练日志**。没有把原始工程中的旧模型变体、消融试验和绘图材料一起搬入。

## 从哪里开始

| 路径 | 内容 |
|---|---|
| `run.py` | 统一运行入口 |
| `src/train_tri_modal_swin_fusion_v3_2.py` | CKESwin 主模型、损失函数与训练循环 |
| `src/FusionDataset.py`、`src/util/` | 图像、图结构、同标本质谱配对和划分 |
| `src/dograph.py`、`src/create_graph/` | YOLO/SAM 导管提取、图构建和图缓存加载 |
| `src/chemistry/` | 提取的质谱 RF 辅助函数及新整理的训练/概率导出入口 |
| `src/evaluation/visual.py` | CKESwin 测试集评估 |
| `src/evaluation/fusion.py` | 21 参数分数融合头的拟合与测试 |
| `configs/` | 便携路径、论文视觉训练配置和 RF 参数 |
| `docs/` | 输入约定、论文对应关系、必要改动与来源哈希 |
| `tests/`、`tools/` | 合成计算检查和纯代码目录检查 |

保留了主模型历史模块名，避免动态导入和 checkpoint 对接时出现不必要的变化。文件名中的 `tri_modal` 指图、全局图像和区域图像三个视觉表征；质谱通过后续融合头接入。

## 运行

先安装与机器匹配的 PyTorch/torchvision，再安装 `requirements.txt`。从原图构图时另装 `requirements-anatomy.txt`；使用原 RF 贝叶斯搜索时另装 `requirements-search.txt`。

在本目录下执行，先修改 `configs/paths.json` 和 `configs/visual.json` 中的外部路径：

```sh
python run.py build-graphs
python run.py train-rf --csv inputs/spectra_20mmu_3percent.csv --test-list inputs/splits/test.txt --output outputs/ms_rf
python run.py train --config configs/visual.json
python run.py evaluate --run_dir outputs/ckeswin --weights best.pt
python run.py fuse --run_dir outputs/ckeswin --weights best.pt --rf_csv outputs/ms_rf/ms_probabilities.csv --stacking_mode diag
```

RF 导出后，先核对概率列顺序与视觉类别顺序，再把 `visual.json` 的 `rf_csv` 指向导出的 CSV。已有图缓存时可跳过构图。运行会生成新结果和权重，但本交付目录没有包含任何既有结果和权重。

**划分字段有一个历史命名约定：`test_file` 实际指验证集 `val.txt`，真正的测试集名单是同目录下的 `test.txt`。** 不要将真正测试名单直接填入这个字段。

默认 RF 入口使用论文固定参数；`--search bayes`/`--search random` 可重跑原代码的 RF 参数搜索。300 轮视觉训练、辅助损失退火参数等按已归档运行配置恢复，不采用后来被改动的脚本默认值。

## 验证与边界

```sh
python tools/check_package.py
python -B tests/test_core.py -v
python -B tests/test_model_smoke.py -v
python -B tests/test_rf_adapter.py -v
```

完整输入要求见 [INPUTS.md](docs/INPUTS.md)。论文对应与需核实之处见 [PAPER_MAPPING.md](docs/PAPER_MAPPING.md)。原代码与本包的区别见 [CHANGES.md](docs/CHANGES.md)。

原始质谱分箱算法没有找到，因此本包从已经过 20 mmu/3% 处理的特征表开始。主工程代码有明确对应链路，但不据此声称历史训练逐字节复现。论文中的双 GPU 记录、triplet 用词及多谱配对的细节均已在对应说明中单列，未擅自改算法以贴合文字。

代码采用 MIT License，版权主体为 zwh。论文 DOI 尚未提供，因此没有填写 DOI；第三方模型权重不随本仓库发布，使用者应遵守其自行获取的依赖与 checkpoint 所适用的许可条款。
