# DoRA 论文复现报告

## 📁 文件结构

```
docs/
├── dora_report.tex          # 主报告文件
├── dora_report.cls          # 自定义文档类（页面格式、代码样式等）
├── cover.tex                # 封面模板
├── README.md                # 本文件
└── images/                  # 图片资源文件夹
    └── 校名.png             # 学校 Logo（可选）
```

## 🔧 编译方法

### 方法 1：命令行编译

```bash
# Windows
cd d:\llm_deploy\LLM\Final-Project\docs
xelatex dora_report.tex
xelatex dora_report.tex  # 编译两次以生成目录

# Linux/macOS
cd /path/to/LLM/Final-Project/docs
xelatex dora_report.tex
xelatex dora_report.tex
```

### 方法 2：VS Code + LaTeX Workshop

1. 安装 LaTeX Workshop 插件
2. 打开 `dora_report.tex`
3. 按 `Ctrl+Alt+B` 编译
4. 按 `Ctrl+Alt+V` 预览 PDF

### 方法 3：Overleaf

1. 将 `docs/` 文件夹上传到 Overleaf
2. 设置编译器为 XeLaTeX
3. 点击 "Recompile" 按钮

## ✏️ 使用说明

### 1. 修改个人信息

打开 `cover.tex`，找到以下部分并替换：

```latex
\sffamily\fontsize{15}{18}\selectfont 姓\quad\quad 名:  & <你的姓名>  \\
\sffamily\fontsize{15}{18}\selectfont 学\quad\quad 号:  & <你的学号> \\
```

### 2. 填写实验结果

打开 `dora_report.tex`，找到第 5 节"可复现性"，填写训练结果：

```latex
\textbf{本次复现} & 4.7M & \textbf{<填写你的BoolQ准确率>} & <填写实际训练时间> \\
```

### 3. 插入论文翻译

报告第 2.2 节预留了 10 页空间用于插入论文翻译：

**方法 A：编译后合并 PDF**
1. 编译 `dora_report.tex` 生成初始 PDF
2. 将你的翻译 PDF 插入到第 5-14 页（使用 Adobe Acrobat 或 PDFtk）

**方法 B：直接嵌入 LaTeX**
1. 删除 `dora_report.tex` 中的 10 个 `\newpage \phantom{.}`
2. 将翻译内容粘贴到第 2.2 节

### 4. 添加校徽（可选）

如果需要在封面显示校徽：

1. 将校徽图片保存为 `images/校名.png`
2. 打开 `cover.tex`，取消以下行的注释：
   ```latex
   \makebox[\textwidth][c]{\includegraphics[width=1.2\textwidth]{images/校名.png}}
   ```

## 📊 报告内容结构

1. **检索和选题**（1页）
   - 论文基本信息
   - 选题理由

2. **阅读和翻译**（11页）
   - 全文精读说明
   - 中文翻译（预留10页）

3. **总结**（8页）
   - 问题背景
   - 研究目标
   - 方法（含数学公式）
   - 数据介绍
   - 实验步骤与结果
   - 结论

4. **批判性分析**（4页）
   - 创新点
   - 不足
   - 延伸实验建议

5. **可复现性**（5页）
   - 环境准备
   - 数据集下载
   - 训练流程
   - 准确率计算

6. **总结与展望**（1页）

## 🛠️ 依赖环境

### LaTeX 发行版

- **Windows**: TeX Live 2023+ 或 MiKTeX
- **Linux**: TeX Live 2023+
- **macOS**: MacTeX 2023+

### 必需宏包

- ctex（中文支持）
- amsmath（数学公式）
- listings（代码高亮）
- booktabs（三线表）
- hyperref（超链接）

**检查宏包是否安装**：
```bash
kpsewhich ctex.sty
kpsewhich listings.sty
```

## 🐛 常见问题

### 1. 编译错误：`! Package fontspec Error: The font "Microsoft YaHei" cannot be found.`

**解决方法**：
- Windows: 确保已安装微软雅黑字体
- Linux: 安装字体 `sudo apt install fonts-wqy-microhei`
- 或修改 `cover.tex`，将 `\fontspec{Microsoft YaHei}` 改为 `\heiti`

### 2. 中文显示异常

确保使用 **XeLaTeX** 编译器（不是 pdfLaTeX）：
```bash
xelatex dora_report.tex
```

### 3. 表格/图片位置偏移

在表格/图片环境中使用 `[H]` 参数强制定位：
```latex
\begin{table}[H]
\begin{figure}[H]
```

### 4. 代码高亮不显示

检查 `listings` 宏包是否正确加载，并在 cls 文件中配置：
```latex
\lstset{
    basicstyle=\ttfamily\small,
    language=bash
}
```

## 📝 自定义设置

### 修改页边距

编辑 `dora_report.cls`：
```latex
\RequirePackage[left=3.18cm,right=3.18cm,top=2.54cm,bottom=2.54cm]{geometry}
```

### 修改行距

编辑 `dora_report.cls`：
```latex
\renewcommand*{\baselinestretch}{1.38}  % 1.38 倍行距
```

### 修改代码样式

编辑 `dora_report.cls` 的 `\lstset` 部分：
```latex
\lstset{
    basicstyle=\ttfamily\footnotesize,  % 字体大小
    backgroundcolor=\color{gray!10},     # 背景颜色
    keywordstyle=\color{blue}            # 关键字颜色
}
```

## 📧 技术支持

如遇到问题，请检查：
1. LaTeX 发行版版本（建议 2023 及以上）
2. 编译器设置（必须使用 XeLaTeX）
3. 中文字体是否安装（微软雅黑、宋体）

## 📄 许可证

本模板基于 MIT 许可证开源，可自由修改和分发。
