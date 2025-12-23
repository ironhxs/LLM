\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{llm_experiment_report}[2025/12/23 LLM Experiment Report]

\LoadClass[12pt,a4paper]{article}

\RequirePackage[T1]{fontenc}
\RequirePackage[left=3.18cm,right=3.18cm,top=2.54cm,bottom=2.54cm]{geometry}
\RequirePackage{url}
\RequirePackage[heading=true]{ctex}
\RequirePackage{graphics}
\RequirePackage{graphicx}
\RequirePackage{array}
\RequirePackage{listings}
\RequirePackage{framed}
\RequirePackage{calc}
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
\setdescription{itemsep=0pt,partopsep=0pt,parsep=\parskip,topsep=2pt}

% 自定义页面风格
\RequirePackage{fancyhdr}
\fancypagestyle{bianyi}{
	\fancyhead{}
	\renewcommand\headrulewidth{0pt}
	\fancyfoot{}
	\fancyfoot[R]{\songti\zihao{5}\thepage}
}

% 设置粗体
\setCJKfamilyfont{song}[AutoFakeBold]{SimSun}
\newcommand*{\bfsongti}{\CJKfamily{song}}
\setCJKfamilyfont{hei}[AutoFakeBold]{SimHei}
\newcommand*{\bfheiti}{\CJKfamily{hei}}
% 设置字体
\setmainfont{Times New Roman}

% 自定义标题样式
\ctexset{
	section={
		format+=\bfsongti\zihao{4}\raggedright,
		name={,、},
		number=\chinese{section},
		aftername=\hspace{0pt},
	},
	subsection={
		format+=\songti\zihao{-4}\raggedright,
		name={,\quad},
		number=\arabic{section}.\arabic{subsection},
		aftername=\hspace{1pt},
	},
	subsubsection={
		format+=\songti\zihao{-4}\raggedright,
		name={,\quad},
		number=\arabic{section}.\arabic{subsection}.\arabic{subsubsection},
		aftername=\hspace{1pt},
	}
}

% 设置浮动体的标题
\RequirePackage{caption}
\DeclareCaptionFont{song}{\songti}
\DeclareCaptionFont{minusfive}{\zihao{5}}
\captionsetup[figure]{%
	format=hang,
	labelsep=quad,
	font={song,minusfive},
	position=bottom
}
\DeclareCaptionFont{minusfour}{\zihao{-4}}
\DeclareCaptionFont{bfsong}{\bfsongti}
\captionsetup[table]{%
	format=hang,
	labelsep=quad,
	font={bfsong,minusfour},
	position=top
}

% 设置代码环境
\RequirePackage[titles]{tocloft}
\renewcommand{\cftdot}{$\cdot$}
\renewcommand{\cftsecdotsep}{1.5}
\setlength{\cftbeforesecskip}{7pt}
\setlength{\cftbeforesubsecskip}{3pt}
\renewcommand{\cftsecfont}{\bfseries\zihao{-4}\songti}
\renewcommand{\cftsecleader}{\cftdotfill{\cftsecdotsep}}
\renewcommand{\cftsecaftersnumb}{\hskip.4em}

\definecolor{dkgreen}{rgb}{0,0.6,0}
\definecolor{gray}{rgb}{0.5,0.5,0.5}
\definecolor{mauve}{rgb}{0.58,0,0.82}

\lstset{
	frame=tb,
	aboveskip=3mm,
	belowskip=3mm,
	showstringspaces=false,
	columns=flexible,
	framerule=1pt,
	rulecolor=\color{gray!35},
	backgroundcolor=\color{gray!5},
	basicstyle={\small\ttfamily},
	numbers=none,
	numberstyle=\tiny\color{gray},
	keywordstyle=\color{blue},
	commentstyle=\color{dkgreen},
	stringstyle=\color{mauve},
	breaklines=true,
	breakatwhitespace=true,
	tabsize=3,
}
