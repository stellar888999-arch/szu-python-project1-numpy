# 小项目 1：NumPy 图像特征提取

仓库地址：[stellar888999-arch/szu-python-project1-numpy](https://github.com/stellar888999-arch/szu-python-project1-numpy)。

用基础 NumPy 补全课程提供的 OpenCLIP ViT-B-32 视觉前向。图像经预处理形成 FP32 [B,3,224,224]；输出 [B,512] 的单位特征向量，保存为包含 ids 与 output_tokens 的 NPZ。内部 [B,50,768] token 序列不等于最终特征。无需 GPU、训练、PyTorch 或 OpenCLIP 软件包。

## 环境与数据

- Python 3.9+；实测 Windows / Python 3.12.10 / NumPy 2.2.6 / Pillow 12.3.0 / imageio 2.37.4。imageio 仅用于课程演示。
- assets/ 包含课程随包的 10 张图、来源及顺序清单、独立 FP32 参考输出。参考数组只用于验证。
- 权重从畅课的 **project1-学生版.zip** 获取。解压到独立目录，将其中 **project1-numpy-student/weights/image_encoder.npz** 复制到本仓库的 **weights/image_encoder.npz**。文件约 311 MiB，保留本地、不进 Git；check.py 的 assets 阶段会核对权重与数据指纹，具体 SHA-256 见 weights/manifest.json。
- 课程演示、讲义映射及练习 PDF 随原始框架保留，本次任务完成对象为 image_encoder.py 的 6 组填空。

## 运行

在仓库根目录运行（PowerShell）：

~~~powershell
python -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '4'
python check.py --stage all
python run.py --output evidence/output_tokens.npz
python evidence/inspect_output.py
python run.py --images assets/images/image_02.jpg assets/images/image_01.jpg --output output_tokens.npz
~~~

自选图片按命令行顺序处理；不指定图片时使用随包的 10 张图。check.py 的 --stage 支持 assets、preprocess、patches、tokens、block0、last_tokens、output 和 batch，可逐层定位问题。

## 文件结构

- **image_encoder.py**：6 组、10 处实现。切块、拆头与合头、注意力投影、两条残差和 MLP、加 CLS 与位置向量、末层 CLS 池化与投影。
- **preprocess_image.py / helpers.py**：课程提供的预处理、LayerNorm、稳定 softmax、erf GELU 和单位化。
- **runtime.py / run.py / check.py**：课程提供的读取、导出与检查入口。
- **config.json / weights/manifest.json / assets/**：架构配置、权重指纹、图片与独立参考。
- **evidence/**：检查日志、生成的向量、导出核验和错误切块反例。
- **report/**：LaTeX 源码；进入该目录用 XeLaTeX 编译两次或运行 Tectonic。见 [实验报告 PDF](output/pdf/project1-report.pdf)。
- **lecture-demos/ / practice/**：随包课堂材料。

## 验证结果

课程检查 8 个阶段全部通过。受检预处理和切块与参考完全相同；10 张图最终特征最大绝对误差 5.662e-7，平均绝对误差 8.688e-8，最大单位范数偏差 1.192e-7。换序与重复输入检查通过。直接 reshape 的形状虽正确，但首图第一块有 3002/3072 个值不同，不能仅用形状验证像素顺序。

这些结果说明给定权重和数据的前向数值复现，不代表下游分类准确率或检索能力。Git 历史保留原始框架和每处填空的真实提交。仓库已公开，助教与老师可直接访问，无需额外邀请。
