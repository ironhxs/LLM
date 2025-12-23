\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{dora_report}[2025/12/23 DoRA Reproduction Report]

\LoadClass[12pt,a4paper]{article}

\RequirePackage[T1]{fontenc}
\RequirePackage[left=3.18cm,right=3.18cm,top=2.54cm,bottom=2.54cm,headheight=14.5pt]{geometry}
\RequirePackage{url}
\RequirePackage[heading=true]{ctex}
\RequirePackage{graphics}
\RequirePackage{graphicx}
\RequirePackage{array}
\RequirePackage{listings}
\RequirePackage{framed}
\RequirePackage{calc}
\RequirePackage{pdfpages}
% 数学宏包
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{bm}
\RequirePackage{titletoc}
% 首行缩进
\RequirePackage{indentfirst}
\renewcommand*{\baselinestretch}{1.38}%行间距
% 修改tabular环境
\let\mcm@oldtabular\tabular
\let\mcm@endoldtabular\endtabular
\renewenvironment{tabular}%
{\bgroup%
	\renewcommand{\arraystretch}{1.5}%
	\mcm@oldtabular}%
{\mcm@endoldtabular\egroup}
% 每行缩进两个汉字
\setlength\parindent{2em}

% 设置颜色
\RequirePackage{xcolor}
% 插入图片
\RequirePackage{float}
% 表格
\RequirePackage{longtable}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage{bigstrut}
\RequirePackage{bigdelim}
% 保护脆落命令
\RequirePackage{cprotect}
\RequirePackage{enumitem}
\setenumerate[1]{itemsep=0pt,partopsep=0pt,parsep=\parskip,topsep=2pt}
\setitemize[1]{itemsep=0pt,partopsep=0pt,parsep=\parskip,topsep=2pt}

% 代码样式
\RequirePackage{listings}
\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    numbers=left,
    numberstyle=\tiny\color{gray},
    backgroundcolor=\color{gray!10},
    keywordstyle=\color{blue},
    commentstyle=\color{green!50!black},
    stringstyle=\color{red},
    showstringspaces=false,
    tabsize=4
}

% 超链接
\RequirePackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=black,
    urlcolor=blue,
    citecolor=blue
}

% 页眉页脚
\RequirePackage{fancyhdr}
\newcommand{\makeheadrule}{%
	\makebox[0pt][l]{\rule[0.55\baselineskip]{\headwidth}{0.4pt}}%
	\rule[0.7\baselineskip]{\headwidth}{0.4pt}}
\renewcommand{\headrule}{%
	{\if@fancyplain\let\headrulewidth\plainheadrulewidth\fi
		\makeheadrule}}
\makeatother
\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{\small DoRA 论文复现报告}
\fancyfoot[C]{\small 第~\thepage~页}

% 定义页面风格
\fancypagestyle{bianyi}{
	\fancyhf{}
	\renewcommand{\headrulewidth}{0pt}
}
