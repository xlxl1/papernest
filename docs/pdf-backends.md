# PDF 解析后端与 OCR 配置

PaperNest 默认使用 PyMuPDF 的文本层和几何规则，速度快且不需要模型。它适合 born-digital PDF；
扫描件、公式密集页、复杂跨栏表格应切换到外部解析器生成结构化 block，再交给 PaperNest 的检索和审计层。

本地上传和 OA 下载的整份扫描 PDF 都支持 OCR 补救；混合 PDF 只对没有文本层的页面补 OCR，保留其他页面的章节结构。
OCR 结果会标记为“非文本层”，引用时应回看原 PDF。

## 公式

普通 OCR 只能提供文字，不能可靠生成 LaTeX。当前插件接口是：

```python
def recognize_formula(png_bytes: bytes, item: dict) -> dict:
    return {"latex": r"\\frac{a}{b}", "confidence": 0.95}
```

在 `.env` 中设置 `PAPERNEST_MATH_OCR_MODULE=your_module`。没有插件时系统保留公式原始文本和坐标，
状态为 `unresolved`，不会伪造 LaTeX。扫描 PDF 没有可靠公式 bbox，建议用 Marker balanced 或 MinerU
先做页面版面分析。

## 无图注图片

带图题的矢量图按图题定位；无图题的嵌入位图按 `get_image_info()` 的 bbox 建立 `uncaptioned` figure 块，
先以“第 N 页，图像 M”占位，运行 `figure-summary --apply` 后会把视觉摘要贴回同一块。纯矢量且无图题的图元
没有可用 image bbox，需要 Marker/MinerU 的 layout 模型。

## 表格

- 有明确 `Table 1` / `表 1` 图题的两列表格会通过额外的数据形态检查；参考文献和正文对齐段不会放宽。
- `detect_tables(..., merge_pages=True)` 会合并页底/页顶且列结构相符的续表，保留 `page_bboxes`、`end_page`。
- `None` 单元格不再被改写成空字符串，会记录 `merged_cells` 和 `needs_review=true`；复杂 rowspan/colspan 仍需人工复核。

## 可选外部解析器

- [Marker](https://github.com/datalab-to/marker)：`marker_single input.pdf --mode balanced --output_format json`
  适合公式、多栏和图表综合解析。
- [MinerU](https://github.com/opendatalab/MinerU)：`mineru-kit parse input.pdf -o output.md --tier standard`
  适合扫描件、复杂版面和表格，支持 CPU/GPU 多级模型。

外部解析器只负责输出 `page_no/block_type/text/bbox/rows/confidence`，PaperNest 继续负责 chunks、向量检索、
引用核验和审计。不要把普通 OCR 当作数学 OCR；模型输出的表格结构要保留置信度，低置信度块显示人工复核标记。
