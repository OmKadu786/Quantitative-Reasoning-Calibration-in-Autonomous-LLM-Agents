from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

OUT = Path('llm_calibration_internship_presentation.pptx')
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

NAVY = RGBColor(13, 27, 42)
BLUE = RGBColor(31, 97, 141)
TEAL = RGBColor(0, 121, 140)
LIGHT = RGBColor(239, 245, 247)
WHITE = RGBColor(255, 255, 255)
DARK = RGBColor(31, 41, 55)
MUTED = RGBColor(91, 105, 116)
GREEN = RGBColor(33, 130, 93)
ORANGE = RGBColor(199, 112, 32)


def add_bg(slide, title=None, section=None):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = LIGHT
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.18))
    bar.fill.solid(); bar.fill.fore_color.rgb = TEAL; bar.line.fill.background()
    if section:
        add_text(slide, section.upper(), 0.55, 0.35, 3.5, 0.22, 9, TEAL, bold=True)
    if title:
        add_text(slide, title, 0.55, 0.64, 12.2, 0.55, 26, NAVY, bold=True)


def add_text(slide, text, x, y, w, h, size=18, color=DARK, bold=False, align=None, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True; tf.vertical_anchor = valign
    p = tf.paragraphs[0]; p.text = str(text); p.font.name = 'Aptos'; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
    if align is not None: p.alignment = align
    return box


def add_bullets(slide, items, x, y, w, h, size=17, color=DARK, gap=5):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = item; p.level = 0; p.font.name = 'Aptos'; p.font.size = Pt(size); p.font.color.rgb = color; p.space_after = Pt(gap)
        p.text = '• ' + p.text
    return box


def add_card(slide, title, body, x, y, w, h, accent=BLUE, body_size=15):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    card.fill.solid(); card.fill.fore_color.rgb = WHITE; card.line.color.rgb = RGBColor(215, 225, 230)
    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.08), Inches(h))
    stripe.fill.solid(); stripe.fill.fore_color.rgb = accent; stripe.line.fill.background()
    add_text(slide, title, x+0.25, y+0.18, w-0.45, 0.35, 17, NAVY, bold=True)
    add_text(slide, body, x+0.25, y+0.65, w-0.45, h-0.8, body_size, DARK)


def add_table(slide, headers, rows, x, y, w, h, font=10):
    shape = slide.shapes.add_table(len(rows)+1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
    table = shape.table
    for j, head in enumerate(headers):
        cell = table.cell(0,j); cell.text = head; cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.name='Aptos'; p.font.size=Pt(font); p.font.bold=True; p.font.color.rgb=WHITE
    for i,row in enumerate(rows,1):
        for j,val in enumerate(row):
            cell=table.cell(i,j); cell.text=str(val); cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if i%2 else RGBColor(230,239,242)
            for p in cell.text_frame.paragraphs:
                p.font.name='Aptos'; p.font.size=Pt(font); p.font.color.rgb=DARK
    widths = [w/len(headers)]*len(headers)
    for j, width in enumerate(widths): table.columns[j].width=Inches(width)
    return table

# 1 Title
slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid(); slide.background.fill.fore_color.rgb = NAVY
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(0.8), Inches(0.14), Inches(5.8))
shape.fill.solid(); shape.fill.fore_color.rgb = TEAL; shape.line.fill.background()
add_text(slide, 'Measuring Quantitative Calibration\nin LLM-Guided ML Optimization', 1.15, 1.25, 10.8, 1.55, 34, WHITE, bold=True)
add_text(slide, 'Internship Research Presentation', 1.18, 3.15, 7, 0.4, 20, RGBColor(185, 215, 221))
add_text(slide, 'QuaRCAA | Credit Fraud and ECG Arrhythmia Pipelines', 1.18, 3.75, 9, 0.35, 16, RGBColor(210, 225, 230))
add_text(slide, 'September 2026', 1.18, 6.45, 4, 0.3, 14, RGBColor(185, 215, 221))

# 2 Contents
slide=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(slide,'Table of Contents','Overview')
contents=[('01','Abstract'),('02','Introduction'),('03','Problem Statement'),('04','Aims and Objectives'),('05','Methodology and Experiments'),('06','Results and Discussion'),('07','Limitations and Status'),('08','References')]
for i,(num,label) in enumerate(contents):
    col=i//4; row=i%4
    add_card(slide, num, label, 0.8+col*6.1, 1.45+row*1.22, 5.35, 0.85, TEAL if col==0 else BLUE, 16)

# 3 Abstract
slide=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(slide,'Abstract','01')
add_card(slide,'Research question','Can calibration awareness and feedback make LLM predictions about ML optimization more accurate and less overconfident?',0.8,1.45,5.8,1.35,TEAL,18)
add_card(slide,'Design','Three LLMs, two imbalanced classification datasets, and three conditions: baseline optimization (C1), calibration warning (C2), and prediction feedback (C3).',6.85,1.45,5.7,1.35,BLUE,17)
add_card(slide,'Main finding','C2 and C3 generally improve calibration and reduce overconfidence, but these gains do not reliably improve downstream ML performance.',0.8,3.2,11.75,1.35,GREEN,18)
add_text(slide,'Core contribution: separating optimization quality from quantitative forecast calibration.',1.0,5.35,11.2,0.55,24,NAVY,bold=True,align=PP_ALIGN.CENTER)

# 4 Introduction
slide=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(slide,'Introduction','02')
add_bullets(slide,[
 'LLMs are increasingly used as agents that propose ML experiments and hyperparameter changes.',
 'A useful agent must do two things: choose actions and estimate the consequences of those actions.',
 'Confidence can be misleading when an agent predicts precise improvements that do not occur.',
 'This project evaluates quantitative predictions against real 3-seed pipeline outcomes.'
],0.9,1.5,11.6,3.2,22)
add_card(slide,'Key distinction','Optimization performance asks: “Did the proposed parameters improve the pipeline?”\n\nCalibration asks: “Did the LLM accurately predict what would happen?”',2.0,5.0,9.3,1.25,TEAL,19)

# 5 Problem
slide=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(slide,'Problem Statement','03')
add_text(slide,'LLM-guided optimization can produce confident numerical claims without reliable evidence.',0.85,1.35,11.7,0.7,27,NAVY,bold=True,align=PP_ALIGN.CENTER)
add_card(slide,'Risk','Overconfident forecasts may lead users to trust weak parameter changes or skip necessary validation.',0.8,2.55,3.8,2.05,ORANGE,18)
add_card(slide,'Measurement gap','Most AutoML-agent evaluations focus on final model performance, not whether the agent correctly predicted the outcome.',4.78,2.55,3.8,2.05,BLUE,18)
add_card(slide,'Research gap','It is unclear whether calibration awareness improves only confidence behavior or also improves optimization decisions.',8.76,2.55,3.8,2.05,TEAL,18)

# 6 Aims
slide=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(slide,'Aims and Objectives','04')
add_bullets(slide,[
 'Measure quantitative calibration in LLM-guided ML optimization.',
 'Compare standard optimization (C1) with calibration awareness (C2).',
 'Test feedback-based self-correction (C3).',
 'Measure downstream task performance separately from calibration quality.',
 'Compare behavior across DeepSeek, OpenAI, and Claude on Credit and ECG tasks.'
],1.0,1.45,11.2,4.0,21)
add_text(slide,'Primary outcome: whether calibration improves without harming optimization.',1.0,5.9,11.2,0.45,20,TEAL,bold=True,align=PP_ALIGN.CENTER)

# 7 Methodology
slide=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(slide,'Methodology and Experiments','05')
steps=[('1','Prompt','LLM receives baseline and previous history.'),('2','Propose','LLM changes parameters and predicts metric ranges.'),('3','Execute','Pipeline evaluates proposed parameters across 3 seeds.'),('4','Compare','Actual means are compared with predictions.'),('5','Diagnose','MACE, RMACE, direction, signal, and overconfidence are recorded.')]
for i,(n,t,b) in enumerate(steps): add_card(slide,n,t+'\n'+b,0.75+(i%3)*4.15,1.45+(i//3)*2.3,3.75,1.65,[TEAL,BLUE,GREEN,ORANGE,TEAL][i],15)
add_text(slide,'3 independent runs × 15 iterations = 45 observations per condition',1.0,6.25,11.2,0.4,18,NAVY,bold=True,align=PP_ALIGN.CENTER)

# 8 Conditions/metrics
slide=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(slide,'Conditions and Metrics','05')
add_table(slide,['Condition','LLM receives','Purpose'],[
 ['C1','Normal history and actual outcomes','Baseline behavior'],
 ['C2','C1 + calibration-audit warning','Test awareness effect'],
 ['C3','C2 + previous prediction feedback','Test self-correction'],
],0.8,1.35,11.75,2.25,15)
add_card(slide,'Metrics','MACE/RMACE: calibration error\nOverconfidence: missed interval on confident side\nMacro F1: downstream ML performance\nSignal accuracy: direction only when change exceeds seed noise',2.0,4.25,9.3,1.7,TEAL,17)

# 9 Credit results
slide=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(slide,'Results: Credit Fraud','06')
add_table(slide,['Model','Cond.','Macro F1','Fraud recall','MACE','RMACE','Overconf.'],[
 ['DeepSeek','C1','0.9250','0.7812','0.0080','2.1654','16.0%'],
 ['DeepSeek','C2','0.9249','0.7840','0.0049','1.4264','1.8%'],
 ['DeepSeek','C3','0.9258','0.7811','0.0060','1.3465','1.3%'],
 ['OpenAI','C1','0.9241','0.7753','0.0088','3.2331','8.0%'],
 ['OpenAI','C2','0.9226','0.7701','0.0044','2.1514','4.0%'],
 ['OpenAI','C3','0.9212','0.7664','0.0040','2.1847','4.9%'],
 ['Claude','C1','0.9273','0.7923','0.0049','1.4238','19.6%'],
 ['Claude','C2','0.9257','0.7875','0.0047','1.0929','8.9%'],
 ['Claude','C3','0.9254','0.7895','0.0040','0.9987','2.2%'],
],0.45,1.25,12.45,4.65,9)
add_text(slide,'Credit: C3 gives the best RMACE/overconfidence for Claude and DeepSeek, but task gains are small.',0.8,6.35,11.8,0.45,15,NAVY,bold=True,align=PP_ALIGN.CENTER)

# 10 ECG results
slide=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(slide,'Results: ECG Arrhythmia','06')
add_table(slide,['Model','Cond.','Macro F1','Macro recall','MACE','RMACE','Overconf.'],[
 ['DeepSeek','C1','0.6270','0.6889','0.0179','1.7397','28.1%'],
 ['DeepSeek','C2','0.6142','0.6750','0.0132','1.6109','7.2%'],
 ['DeepSeek','C3','0.6082','0.6630','0.0103','0.8248','5.2%'],
 ['OpenAI','C1','0.5873','0.6525','0.0194','2.6815','24.2%'],
 ['OpenAI','C2','0.5849','0.6613','0.0087','2.3514','16.3%'],
 ['OpenAI','C3','0.5810','0.6447','0.0087','2.2716','8.9%'],
 ['Claude','C1','0.6247','0.7337','0.0099','1.1586','20.7%'],
 ['Claude','C2','0.6067','0.6959','0.0107','1.1328','12.8%'],
 ['Claude','C3','0.6066','0.6970','0.0127','1.1710','11.6%'],
],0.45,1.25,12.45,4.65,9)
add_text(slide,'ECG: C3 strongly improves calibration for DeepSeek, but does not recover C1 task performance.',0.8,6.35,11.8,0.45,15,NAVY,bold=True,align=PP_ALIGN.CENTER)

# 11 Discussion
slide=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(slide,'Results and Discussion','06')
add_card(slide,'Finding 1','C2 and C3 generally reduce overconfidence and calibration error.',0.8,1.35,3.8,1.65,TEAL,18)
add_card(slide,'Finding 2','Better calibration does not reliably produce better ML optimization results.',4.78,1.35,3.8,1.65,ORANGE,18)
add_card(slide,'Finding 3','Model differences depend on the dataset; no LLM dominates every task.',8.76,1.35,3.8,1.65,BLUE,18)
add_text(slide,'Core interpretation',0.9,3.75,11.4,0.4,20,NAVY,bold=True,align=PP_ALIGN.CENTER)
add_text(slide,'“Knowing” the likely outcome and “doing” the best optimization are separable capabilities.',1.5,4.35,10.3,0.9,28,TEAL,bold=True,align=PP_ALIGN.CENTER)
add_text(slide,'C3 is implemented as feedback-based self-correction; it is not the deleted parameter-clamping guard.',1.2,5.75,10.9,0.5,16,MUTED,align=PP_ALIGN.CENTER)

# 12 Limitations
slide=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(slide,'Limitations','07')
add_bullets(slide,[
 'Only 3 independent runs per condition; the 45 iterations are sequential, not independent replicates.',
 'Gemini is incomplete and excluded from the validated aggregate.',
 'C2 is an awareness intervention; C3 is a feedback intervention, not proof of internal learning.',
 'The signal-detection threshold and directional overconfidence definition are exploratory.',
 'ECG aggregate macro precision/recall were measured as actual outcomes but were not predicted by the original schema.',
 'Run-level confidence intervals and effect sizes are still needed for stronger claims.'
],0.9,1.35,11.7,4.6,18)

# 13 Status / next
slide=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(slide,'Status and Next Steps','07')
add_card(slide,'Completed','C1, C2, and C3 runs for DeepSeek and OpenAI; C1, C2, and C3 for Claude Credit/ECG; validated artifacts and cost logs.',0.8,1.35,5.7,1.75,GREEN,16)
add_card(slide,'Next analysis','Use the 3 runs as the independent units; report run-level means, effect sizes, confidence intervals, and sensitivity analyses.',6.85,1.35,5.7,1.75,BLUE,16)
add_card(slide,'Research conclusion','Calibration awareness and feedback improve confidence behavior, but do not reliably improve downstream ML performance.',0.8,3.65,11.75,1.45,TEAL,19)
add_text(slide,'Future: feedback-only vs warning-plus-feedback controls; wider-interval control; improved ECG prediction schema.',1.0,5.75,11.3,0.45,16,MUTED,align=PP_ALIGN.CENTER)

# 14 References
slide=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(slide,'References','08')
refs=[
 'Kadavath et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.',
 'Barkan et al. (2024). Metacognition and confidence in agentic language-model workflows. NeurIPS Workshop.',
 'Wang et al. (2026). MIRROR: studying the knowing-doing gap and architectural constraints in LLM agents. arXiv:2604.19809.',
 'Yang et al. (2024). OPRO: Optimization by PROmpting. Google DeepMind.',
 'Zheng et al. (2024). AgentHPO: LLM-based hyperparameter optimization.',
 'Moody et al. (2001). MIT-BIH Arrhythmia Database and ECG classification benchmarks.'
]
add_bullets(slide,refs,0.9,1.4,11.5,4.8,17)
add_text(slide,'Project repository: Quantitative Reasoning Calibration in Autonomous LLM Agents',0.9,6.35,11.5,0.3,13,MUTED,align=PP_ALIGN.CENTER)

# Footer slide numbers
for idx, slide in enumerate(prs.slides, 1):
    add_text(slide, str(idx), 12.45, 7.13, 0.45, 0.2, 9, MUTED, align=PP_ALIGN.RIGHT)

prs.save(OUT)
print(OUT.resolve())
