"""EPSR 投稿字数统计（权威口径）+ 图表落位检查。

期刊口径（Guide for authors, Electric Power Systems Research，2026-02-20 起）:
    "All text including Abstract, tables, figures, and appendixes are counted
     for the word limits ... but references and acknowledgements are not
     counted."
    Full Length Article: max. 7000 words
    "Manuscripts exceeding word limits may be rejected before peer review."

因此计入：正文散文 + Abstract + 图注 + 表格（含表头/表注）；
不计入：References、Acknowledgements。

用法
----
    python tools/wordcount_epsr.py            # 只报字数
    python tools/wordcount_epsr.py --floats   # 附带图表落位检查（需已编译 main.pdf）

两个必须小心的坑（都曾真实踩过）
--------------------------------
1. 必须先剥离 LaTeX 注释再匹配浮动环境。导言区注释里若出现字面的
   begin-figure token，会开出一个"幽灵浮动体"，一路吞到第一个真正的
   \\end{figure}，把整章 Introduction 当图注统计 —— 曾令图注从 257 词虚报为
   1506 词、总数从 6858 虚报为 8107（超限假警报）。
2. caption 必须用平衡括号提取，不能用 \\caption\\{[^}]*\\} 这种正则：图注里
   含 $...$、\\textbf{...}、(a)--(f) 等嵌套括号会被截断。
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'manuscript')
ROOT = os.path.normpath(ROOT)
LIMIT = 7000


# ---------------------------------------------------------------------------
# 文本清洗
# ---------------------------------------------------------------------------
def strip_comments(s: str) -> str:
    """剥离未转义的 % 注释。这一步必须在任何环境匹配之前做（见模块 docstring 坑 1）。"""
    return re.sub(r'(?<!\\)%.*', '', s)


def clean(s: str) -> str:
    s = strip_comments(s)
    s = re.sub(r'\\cite\w*\{[^}]*\}', ' CITE ', s)
    s = re.sub(r'\\ref\{[^}]*\}', ' REF ', s)
    s = re.sub(r'\\label\{[^}]*\}', ' ', s)
    s = re.sub(r'\\(?:textbf|textit|emph|texttt|multicolumn|multirow|underline)'
               r'\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\$[^$]*\$', ' EQ ', s)
    s = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', s)
    s = s.replace('{', ' ').replace('}', ' ').replace('&', ' ')
    s = re.sub(r'[\\_^~]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def wc(s: str) -> int:
    return len(clean(s).split())


def balanced_arg(src: str, cmd: str, start: int = 0) -> str:
    """提取 \\cmd{...} 的平衡括号参数（见模块 docstring 坑 2）。"""
    i = src.find('\\' + cmd, start)
    if i < 0:
        return ''
    j = src.find('{', i)
    if j < 0:
        return ''
    depth, k = 0, j
    while k < len(src):
        if src[k] == '{':
            depth += 1
        elif src[k] == '}':
            depth -= 1
            if depth == 0:
                return src[j + 1:k]
        k += 1
    return ''


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def count() -> dict:
    tex_raw = open(os.path.join(ROOT, 'main.tex'), encoding='utf-8').read()
    tex = strip_comments(tex_raw)          # 关键：先去注释

    FLOAT = re.compile(r'\\begin\{(figure\*?|table\*?)\}(.*?)\\end\{\1\}', re.S)

    # 正文散文：切掉浮动体与 \input（它们单独计）
    start = tex.find(r'\section{Introduction}')
    end = tex.find(r'\bibliographystyle')
    body = tex[start:end]
    prose = FLOAT.sub(' ', body)
    prose = re.sub(r'\\input\{[^}]*\}', ' ', prose)
    body_cons = wc(prose)
    prose_noeq = re.sub(r'\\begin\{(equation|align|gather)\*?\}.*?\\end\{\1\*?\}',
                        ' EQBLOCK ', prose, flags=re.S)
    body_w = wc(prose_noeq)
    neq = len(re.findall(r'\\begin\{(equation|align|gather)\*?\}', prose))

    ab_w = wc(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
                        tex, re.S).group(1))

    # 图注：逐个 float 用平衡括号取 caption
    caps = []
    for m in FLOAT.finditer(tex):
        inner = m.group(2)
        lab = re.search(r'\\label\{([^}]*)\}', inner)
        cap = balanced_arg(inner, 'caption')
        caps.append((lab.group(1) if lab else '?', m.group(1), wc(cap)))
    cap_w = sum(n for _, _, n in caps)

    # 表格：只算 main.tex 真正 \input 的
    tabs = []
    tab_w = 0
    for rel in re.findall(r'\\input\{([^}]*)\}', tex):
        path = os.path.join(ROOT, rel if rel.endswith('.tex') else rel + '.tex')
        if not os.path.exists(path):
            tabs.append((rel, None))
            continue
        t = open(path, encoding='utf-8').read()
        t = re.sub(r'\\includegraphics(\[[^\]]*\])?\{[^}]*\}', ' ', t)
        w = wc(t)
        tab_w += w
        tabs.append((rel, w))

    return dict(body_w=body_w, body_cons=body_cons, neq=neq, ab_w=ab_w,
                caps=caps, cap_w=cap_w, tabs=tabs, tab_w=tab_w)


def report(r: dict) -> int:
    print('included tables (\\input by main.tex):')
    for rel, w in r['tabs']:
        print(f'  {"MISSING" if w is None else w:>7}  {rel}')
    print('\nfigure/table captions:')
    for lab, env, n in r['caps']:
        print(f'  {n:>7}  {lab}  [{env}]')
    print(f'  {r["cap_w"]:>7}  = caption subtotal ({len(r["caps"])} floats)')
    print()
    print(f'{r["body_w"]:6d}  body prose (display equations excluded, {r["neq"]} eqs)')
    print(f'{r["ab_w"]:6d}  abstract')
    print(f'{r["cap_w"]:6d}  captions')
    print(f'{r["tab_w"]:6d}  tables')
    print('-' * 56)
    tot = r['body_w'] + r['ab_w'] + r['cap_w'] + r['tab_w']
    cons = r['body_cons'] + r['ab_w'] + r['cap_w'] + r['tab_w']
    print(f'{tot:6d}  <== REPORTABLE Article Length   {LIMIT - tot:+5d} vs {LIMIT}')
    print(f'{cons:6d}      upper bound, symbols inside display equations also')
    print(f'{"":6s}      counted as words ({LIMIT - cons:+d} vs {LIMIT}) -- not how a')
    print(f'{"":6s}      journal counts; shown only as a safety margin')
    print()
    print('VERDICT:', 'WITHIN LIMIT' if tot <= LIMIT else 'OVER LIMIT',
          f'(declare {tot} in the cover letter)')
    if cons > LIMIT:
        print(f'         margin is thin: only {LIMIT - tot} words of slack, and the'
              f' equation-inclusive\n         upper bound ({cons}) exceeds the cap.')
    return tot


def check_floats() -> None:
    try:
        import fitz
    except ImportError:
        print('\n[floats] pymupdf not available; skipped')
        return
    pdf = os.path.join(ROOT, 'main.pdf')
    if not os.path.exists(pdf):
        print('\n[floats] main.pdf not found; compile first')
        return
    d = fitz.open(pdf)
    ref = None
    cap, cite = {}, {}
    for i in range(d.page_count):
        t = d[i].get_text()
        for m in re.finditer(r'Fig\.\s*(\d+)\.\s+[A-Z]', t):
            cap.setdefault(int(m.group(1)), i + 1)
        for m in re.finditer(r'Fig\.\s*(\d+)(?![\.\d])', t):
            cite.setdefault(int(m.group(1)), i + 1)
        if ref is None and 'References' in t and re.search(r'\[\s*1\s*\]', t):
            ref = i + 1
    print(f'\n[floats] {d.page_count} pages, References start p{ref}')
    bad = 0
    for k in sorted(cap):
        cp, ci = cap[k], cite.get(k)
        in_body = ref is None or cp < ref
        if not in_body:
            bad += 1
        off = f'{cp - ci:+d}' if ci else 'n/a'
        print(f'  Fig.{k:<3} caption p{cp:<4} cite p{str(ci):<5} offset {off:<5}'
              f' {"ok" if in_body else "AFTER REFERENCES!"}')
    print(f'[floats] {len(cap) - bad}/{len(cap)} figures placed in the body')


def per_section() -> None:
    """按 \\section/\\subsection 切分正文散文词数，用于定位压缩目标。"""
    tex = strip_comments(
        open(os.path.join(ROOT, 'main.tex'), encoding='utf-8').read())
    FLOAT = re.compile(r'\\begin\{(figure\*?|table\*?)\}(.*?)\\end\{\1\}', re.S)
    start = tex.find(r'\section{Introduction}')
    end = tex.find(r'\bibliographystyle')
    body = FLOAT.sub(' ', tex[start:end])
    body = re.sub(r'\\input\{[^}]*\}', ' ', body)

    # 必须匹配带星的 \section*{}，否则 CRediT / 声明类小节会被并进上一节，
    # 曾让 Limitations 虚高到 491 词（实际含 ~180 词的声明块）。
    heads = []
    for m in re.finditer(r'\\(section|subsection|subsubsection)(\*?)\{', body):
        title = balanced_arg(body, m.group(1) + m.group(2), m.start())
        if not title:
            j = body.find('{', m.start())
            depth, k = 0, j
            while k < len(body):
                if body[k] == '{':
                    depth += 1
                elif body[k] == '}':
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            title = body[j + 1:k]
        heads.append((m.start(), m.group(1),
                      ('* ' if m.group(2) else '') + title))
    print(f'{"section":58s} words')
    print('-' * 66)
    for i, (pos, lvl, title) in enumerate(heads):
        nxt = heads[i + 1][0] if i + 1 < len(heads) else len(body)
        seg = body[pos:nxt]
        seg = re.sub(r'\\begin\{(equation|align|gather)\*?\}.*?\\end\{\1\*?\}',
                     ' EQBLOCK ', seg, flags=re.S)
        n = wc(seg)
        indent = {'section': '', 'subsection': '  ', 'subsubsection': '    '}[lvl]
        print(f'{indent}{title[:54]:{54 - len(indent)}s}  {n:6d}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--floats', action='store_true',
                    help='additionally verify figure placement in main.pdf')
    ap.add_argument('--sections', action='store_true',
                    help='print per-section prose word counts')
    a = ap.parse_args()
    r = count()
    report(r)
    if a.sections:
        print()
        per_section()
    if a.floats:
        check_floats()
    sys.exit(0)
