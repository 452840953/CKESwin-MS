# CKESwin-MS 主体代码

这份代码对应当前 KBS v5.24 论文的主体方法：**解剖知识引导的视觉表征 → 独立质谱随机森林 → 同标本分数级融合**。

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
| `tests/`、`tools/` | 测试与辅助工具 |

文件名中的 `tri_modal` 指图、全局图像和区域图像三个视觉表征；质谱通过后续融合头接入。

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

RF 导出后，先核对概率列顺序与视觉类别顺序，再把 `visual.json` 的 `rf_csv` 指向导出的 CSV。已有图缓存时可跳过构图。
