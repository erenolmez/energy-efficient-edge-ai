"""Mechanical typography normalization for this report's generated prose."""
import re
from pathlib import Path
root=Path(__file__).resolve().parent
for name in ('progress_report.tex','payback_example.tex','batch1_example.tex','optimized_payback.tex','accuracy_sensitivity.tex','runtime_explanations.tex'):
    p=root/name
    text=p.read_text(encoding='utf-8')
    text='\n'.join(line if line.startswith(('\\definecolor','\\documentclass','\\usepackage','\\input','\\begin{center}\\includegraphics')) else re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',line) for line in text.splitlines())+'\n'
    for token in ('FP32','FP16','FP64','INT8','R101','R50','R34','R18'):
        spaced=re.sub(r'(?<=[A-Za-z])(?=\d)',' ',token)
        text=text.replace(spaced,token)
    text=text.replace('images/batch 8_grouping.png','images/batch8_grouping.png')
    text=text.replace('batch 1_example.tex','batch1_example.tex')
    text=text.replace('GPUJ','GPU J')
    text=re.sub(r'([:;])(?=\d)',r'\1 ',text)
    text=re.sub(r'(\\%,)(?=\d)',r'\1 ',text)
    p.write_text(text,encoding='utf-8')
