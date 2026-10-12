from __future__ import annotations
import base64
import csv
import html
import io
import os
import json
import zipfile
from pathlib import Path
import numpy as np
from .common import *

GLOSSARY={
 'Socket supply state':'ON/OFF labels describe whether a socket supplies power. At a low microwave setting, the magnetron may cycle while the socket remains ON.',
 'Aggregate measurements and multiple labels':'Inputs are the aggregate voltage and current of connected appliances. Each recording uses five physical sockets; their annotations map to 11 appliance-class channels across recordings. Multiple classes may be ON at once.',
 '51.2 kHz, cycles and tokens':'The signals contain about 51,200 samples per second and 1,024 samples per channel in each approximately 20 ms AC cycle. Five complete cycles form a 100 ms token; 256 tokens cover 25.6 seconds.',
 'Active P, apparent S and total non-active N':'P is the cycle mean of voltage times current; S is RMS voltage times RMS current; N is sqrt(S²−P²). For distorted waveforms, N does not represent fundamental reactive power.',
 'F: phase-referenced harmonics':'Current harmonics 1–25 use the fundamental voltage phase as a reference. Features retain real and imaginary components, absolute and relative magnitudes, and backward changes over 5 and 50 cycles.',
 'W1: learned waveform representation':'A trainable convolutional encoder processes phase-aligned voltage, current and non-active current. This representation is computed and learned from aggregate waveforms; it requires no additional meter.',
 'Fryze non-active current':'Subtracting (P/U_rms²)u from total current gives the component complementary to active current. It remains an aggregate waveform and does not isolate an individual appliance.',
 'DVI: directed V–I trajectories':'Voltage is the horizontal coordinate and current the vertical coordinate. Density captures shape, while signed horizontal and vertical changes capture direction. Each of two current types has three channels. Five cycles are deposited separately, without connecting across cycles.',
 'CNN-GRU':'Convolutions extract patterns in neighboring tokens, and a unidirectional gated recurrent unit updates temporal memory. Its state is initialized for each input window and is not carried between windows.',
 'Deep causal convolutions with large kernels':'Large kernels capture longer local patterns. Depthwise convolutions process channels separately, followed by feed-forward channel interaction. Left padding excludes measurements after the output time.',
 'Patched Transformer':'Four tokens form a patch, and attention relates historical patches. This implementation uses a strictly causal mask, unlike the noncausal forecasting protocol in the original paper.',
 'Masks and continuity':'Masks identify valid measurements and labels; an invalid label is not OFF. Continuity marks segments without missing frames, backward timestamps or large gaps. Windows cannot cross breaks or data partitions.',
 'Macro-average AP':'Average Precision (AP) summarizes precision and recall across thresholds. Computable records containing the appliance receive equal weight, followed by equal weighting across the six primary classes, folds and seeds. AP is distinct from classification accuracy.',
 'Recall':'Recall is the fraction of true ON states predicted as ON. The 0.95 calibration target applies to the calibration partition and does not guarantee 95% recall under another recording script.',
 'Precision and F1':'Precision is the fraction of predicted ON states that are truly ON. F1 is the harmonic mean of precision and recall.',
 'OFF false-positive rate':'OFF FPR is the fraction of true OFF states predicted as ON. Absent-class false positives check whether the model predicts activity when that appliance is not connected.',
 'Power-conditioned concordance':'Each positive score is compared with all OFF scores in the same recording and 50 W aggregate-power bin. A higher score receives 1 and a tie 0.5. Positive examples and then recordings receive equal weight. Interpret concordance alongside matching coverage.',
 'Script families and two folds':'Family A contains records 2/4/6 and family B records 3/5/7. The two folds exchange training and evaluation families. Records sharing a script use the same annotation-row boundaries to prevent switching-order leakage. Cross-script comparisons change background, settings and timing together; they do not test another household.',
 'Seeds and independent reruns':'Seeds affect initialization and training order. Multiple seeds describe training variability. An independent rerun fits a model afresh; semantic hashes must match on the same device and environment.',
 'Paired block bootstrap and confidence intervals':'Each recording is divided into 60-second reception-time blocks, sampled with replacement. Both models, all heads and all seeds share the same block indices. Overall metrics such as AP are recomputed after pooling endpoints. The 95% intervals exclude model-selection and threshold-estimation uncertainty.',
 'FP, FN and high-power OFF states':"FP denotes false positives and FN false negatives. The high-power OFF boundary is the 80th percentile of aggregate power during that class's OFF states in the fold's training partition, frozen for hold-out evaluation.",
 'Pretraining and fine-tuning':'The representation encoder is pretrained only on single-appliance recordings of classes and settings seen in the fold. Aggregate training uses a newly initialized temporal backbone. This tests transfer for the same physical devices, without evidence of generalization to new brands.',
 'FP32, MACs and NPU execution':'FP32 denotes single precision. MACs count multiply-accumulate operations; execution time also depends on operators, memory and I/O. NPU performance must be measured on the target hardware.'}

DISPLAY_TEXT = {
    '跨脚本AP提升及CI': 'Cross-script AP gain and confidence interval',
    'Recall非劣': 'Recall non-inferiority',
    'OFF误报保护': 'OFF false-positive safeguard',
    '逐类别Recall保护': 'Per-class recall safeguard',
    '两折seed一致性': 'Seed consistency in both folds',
    '第三脚本缺席类别误报': 'Absent-class false positives in the third script',
    '稳态方向': 'Steady-state direction',
    'Bootstrap有效数量': 'Valid bootstrap replicate count',
    '值得扩大复验（仅限TDHA受控记录上的探索性证据）': 'Expanded replication is supported by exploratory evidence from controlled TDHA recordings',
    '未达到原冻结扩大复验判据；完整保留诊断结果': 'The original frozen criteria for expanded replication were not met. Diagnostic results are retained',
    '典型ON': 'typical ON',
    '低功率ON': 'low-power ON',
    '高功率OFF': 'high-power OFF',
}

def resolve_report_font(font=None, *, required=False):
    """Resolve an optional report font without importing rendering dependencies."""
    font = font or os.environ.get('TDHA_REPORT_FONT')
    if not font:
        require(not required, 'Report export requires --font or TDHA_REPORT_FONT pointing to a licensed CJK TTF')
        return None
    path = Path(font).expanduser().resolve()
    require(path.is_file(), f'Report font file is missing: {path}')
    return path


def validate_report_font(font=None):
    """Validate explicitly requested report resources before starting the queue."""
    path = resolve_report_font(font, required=True)
    from matplotlib.font_manager import FontProperties
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image
    try:
        FontProperties(fname=str(path)).get_name()
        TTFont('TDHAValidation', str(path))
    except Exception as exc:
        raise Blocked(f'Report font cannot be loaded as a TrueType font: {path}: {exc}') from exc
    return path


def report(run, font=None):
    font = validate_report_font(font)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties,fontManager
    from matplotlib.patches import FancyBboxPatch
    run=Path(run);out=run/'report';out.mkdir(parents=True,exist_ok=True);plots=out/'images';plots.mkdir(exist_ok=True)
    fontManager.addfont(str(font));plt.rcParams.update({'font.family':FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False,'axes.facecolor':'#f7fafc','savefig.facecolor':'white','font.size':10})
    sections=[];figures=[]
    def paragraph(title,text):sections.append(('text',title,text))
    def figure(name,title,caption,draw,w=11,h=5):
        fig,ax=plt.subplots(figsize=(w,h));draw(fig,ax);fig.tight_layout();path=plots/(name+'.png');fig.savefig(path,bbox_inches='tight');plt.close(fig)
        sections.append(('figure',title,(path,caption)));figures.append(path)
    completed=(run/'execution_complete.json').exists()
    paragraph('Task and scope','The task predicts appliance supply states from aggregate voltage and current. Each recording has five physical sockets mapped across records to 11 appliance-class channels. Labels describe supply state rather than internal heating, operating settings or appliance power. All seven TDHA records informed exploration. Findings apply to these controlled aggregate recordings; real-household, cross-brand and deployment performance remain untested.')
    anomalies=[a for p in (run/'audit/pretrain_labels').glob('*.json') for a in read(p).get('anomalies',[])]
    if anomalies:
        paragraph('Malformed single-appliance labels',f'The pretraining annotations contain {len(anomalies)} malformed 1)0 entries. The corresponding class is unknown from each malformed entry until the next valid state; ±150 ms and ±500 ms boundary masks are applied at both ends. Sources and timestamps are recorded in audit/pretrain_labels.')
    if (run/'audit/resume_upgrade/label_unknown_v1/receipt.json').exists():
        paragraph('Label-parser version compatibility','After the label-parser update, identity checks covered aggregate caches and models, predictions, thresholds and metrics from 28 primary fits. Compatibility records associate parser versions, capacity measurements and candidate-selection code; see audit/resume_upgrade.')
    if any((run/'audit/operator_preflight_skips').glob('*.json')):
        paragraph('Operator preflight record','This run used --skip-preflight, so synthetic NPU operator checks are recorded as skipped. Dependencies, data, capacity, training numerics and independent-rerun checks are recorded separately; see audit/operator_preflight and audit/operator_preflight_skips.')
    if not completed:
        reason=read(run/'blocker.json').get('reason') if (run/'blocker.json').exists() else 'The NPU task queue is incomplete.'
        paragraph('NPU execution status: partial',f'{reason} This report covers the experiment design and completed tasks. Chart cells remain empty where results are unavailable.')
    else:
        from .analysis import analyze
        decision=read(run/'analysis/decision.json') if (run/'analysis/decision.json').exists() else analyze(run)
        paragraph('Main finding',DISPLAY_TEXT.get(decision['conclusion'], decision['conclusion'])+'. The frozen candidate is '+decision['candidate']['net']+'-'+decision['candidate']['rep']+'. The final comparison uses this prespecified candidate.')
    paragraph('1. Inputs and time resolution','Inputs are synchronized aggregate voltage U and current I, with 1,024 samples per channel per cycle at approximately 51.2 kHz. Physical features or learned encodings use fully received cycles; all five consecutive cycles contribute to a 100 ms token. The model reads 256 historical tokens (25.6 seconds) and produces 11 current logits. The 100 ms interval describes updates, while 25.6 seconds describes history. Label changes do not trigger model execution.')
    def flow(fig,ax):
        ax.set(xlim=(0,11),ylim=(0,3));ax.axis('off')
        boxes=[('Aggregate U / I\n51.2 kHz',0.1,1.6),('Complete cycle\n1,024 samples / 20 ms',2.3,1.6),('Five-cycle token\n100 ms',4.5,1.6),('256-token history\n25.6 s',6.7,1.6),('11 class outputs\nIndependent ON scores',8.9,1.6)]
        for text,x,y in boxes:
            ax.add_patch(FancyBboxPatch((x,y-.4),1.85,.8,boxstyle='round,pad=.05',fc='#e0f2fe',ec='#0284c7'));ax.text(x+.925,y,text,ha='center',va='center',fontsize=10)
        for a,b in zip(boxes,boxes[1:]):ax.annotate('',xy=(b[1]-.08,1.6),xytext=(a[1]+1.95,1.6),arrowprops={'arrowstyle':'->','color':'#334155'})
        ax.text(5.5,.45,'Socket states, appliance lists and settings provide supervision and evaluation only.',ha='center',color='#b45309',fontsize=11)
    figure('01_flow','Aggregate inputs, supervision and outputs','Arrows carry representations derived from aggregate measurements. Appliance lists and socket labels are excluded from the model inputs.',flow,h=3)
    paragraph('2. Labels and data partitions','The 11 output classes are: '+ ', '.join(NAMES)+'. Primary analysis covers the first six classes, which have training positives in both folds; other classes are reported separately. Heads without training positives are marked untrained and receive no recognition metrics. Class-specific supervision excludes ±150 ms around switching events, with an additional ±500 ms analysis. Record 1 is a third script combination and contains positive states. Family A (2/4/6) is partitioned at annotation rows 255 and 340; family B (3/5/7) at rows 236 and 315. Early partitions train the models, middle partitions provide early stopping and frozen thresholds, and later partitions select the candidate. Cross-script predictions follow configuration freezing and independent-rerun checks.')
    paragraph('3. Input representations','P contains six electrical quantities and causal differences or baseline residuals. F adds phase-referenced current harmonics 1–25, relative amplitudes and changes. W1 learns a 16-dimensional cycle signature from phase-aligned voltage, current and non-active current, then aggregates five cycles. DVI forms directed trajectories for total and non-active current, each with density and horizontal and vertical direction channels. Block-RMS normalization captures trajectory shape while the power branch retains absolute magnitudes.')
    if (run/'cache/ready.json').exists():_input_figures(run,figure)
    paragraph('4. Causal temporal networks','N0 is a compact reference: causal convolutions followed by a two-layer unidirectional GRU. N1 uses width 192 and six residual blocks, combining parallel 31-point depthwise and 3-point convolutions with per-token channel interaction. N2 groups four tokens into a patch and relates historical patches using four causal Transformer layers with six attention heads. N1 adapts large-kernel and channel designs from ModernTCN and ConvNeXt; N2 adapts the patching idea from PatchTST. These are causal multi-label adaptations, with differing capacity and front-end costs; the comparison does not isolate network depth.')
    def networks(fig,ax):
        ax.set(xlim=(0,10),ylim=(-.6,3.4));ax.axis('off')
        rows=[('N0','128→32 causal convolution','2-layer forward GRU\nHidden size 48','48→32→11'),('N1','128→192 projection','6 causal large-kernel blocks\nParallel 31/3 + FFN','Final 192→11'),('N2','4 tokens per patch','4 causal attention layers\nWidth 192 / 6 heads','Final 192→11')]
        for k,row in enumerate(rows):
            y=2.7-k*1.2;ax.text(.15,y,row[0],va='center',fontweight='bold')
            for j,text in enumerate(row[1:]):
                x=.9+j*3;ax.add_patch(FancyBboxPatch((x,y-.35),2.4,.7,boxstyle='round,pad=.04',fc=['#e0f2fe','#dcfce7','#fef3c7'][k],ec='#64748b'));ax.text(x+1.2,y,text,ha='center',va='center')
                if j<2:ax.annotate('',xy=(x+2.92,y),xytext=(x+2.46,y),arrowprops={'arrowstyle':'->'})
    figure('networks','Three temporal network architectures','The shared front end produces 128-dimensional tokens. N1 has a theoretical convolutional receptive field longer than the input, but can use only the supplied 25.6-second history. N2 likewise accesses only received patches within the current window.',networks,h=4)
    paragraph('5. Fixed training and selection rules','Stage A compares 3 networks × 3 high-frequency representations × 2 folds at seed 42, plus N0-P power controls: 20 fits. Only the later training-family partition selects a single candidate. Stage B adds seeds 11 and 73 for N0-W1 and the candidate: 8 fits. Representation pretraining and aggregate fine-tuning in both folds add 4 fits; two independent reruns on fold A at seed 42 bring the total to 34. Training uses all eligible natural one-second endpoints, class-balanced BCE, FP32 and AdamW, with at most 40 epochs and patience 5. Each calibratable head selects its highest logit threshold achieving calibration recall ≥0.95. Hold-out results do not change thresholds.')
    tasks=[]
    for path in sorted((run/'tasks').glob('*/done.json')):tasks.append(read(path))
    dump(out/'task_summary.json',tasks)
    table=[]
    for a in tasks:
        row={k:a['task'].get(k) for k in ('id','net','rep','fold','seed','kind')};row.update(parameters=a['parameters'],elapsed_s=a['elapsed_s']);row.update({k:a.get('summary',{}).get(k) for k in ('AP','Recall','OFF_FPR')});table.append(row)
    with (out/'task_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(table[0]) if table else ['id','AP','Recall','OFF_FPR'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(table)
    paragraph('6. Metrics and aggregation','AP measures ranking quality, recall measures missed ON states, and OFF FPR measures false activity predictions. Interpret all three together. Primary metrics weight records equally, then classes, folds and seeds equally. AP/ROC-AUC are unavailable for records lacking the required class support and are not replaced with zero. High-power boundaries use the training OFF-state 80th percentile. Read 50 W conditioned concordance alongside matching coverage. The complete grids below contain Stage A seed-42 results only.')
    paragraph('Supplemental results at a fixed 0.5 threshold','In per-class CSV/JSON files, main_05 rows use logit ≥0 (probability ≥0.5); main rows use individually calibrated, frozen thresholds. The 0.5 results are descriptive and do not select candidates or replace the primary decision. strict500, steady and overlap denote wider switching boundaries, steady states away from switches, and simultaneous appliance activity.')
    for part,label in [('time','Within-script later partition'),('cross','Frozen cross-script evaluation')]:
        for metric,chinese in [('AP','Macro-average AP'),('Recall','Macro-average recall'),('OFF_FPR','Macro-average OFF FPR')]:
            grid=np.full((3,3),np.nan)
            for i,n in enumerate(('N0','N1','N2')):
                for j,r in enumerate(('F','W1','DVI')):
                    vals=[]
                    for fold in ('A','B'):
                        p=run/'tasks'/f'{n}-{r}-{fold}-s42'/f'{part}_metrics.json'
                        if p.exists():
                            v=read(p)['summary'][metric]
                            if v is not None:vals.append(v)
                    if len(vals)==2:grid[i,j]=np.mean(vals)
            def draw_grid(fig,ax,g=grid):
                im=ax.imshow(np.ma.masked_invalid(g),vmin=0,vmax=1,cmap='viridis');fig.colorbar(im,ax=ax,shrink=.7)
                ax.set_xticks(range(3),['F harmonics','W1 waveforms','DVI trajectories']);ax.set_yticks(range(3),['N0 CNN-GRU','N1 large-kernel convolution','N2 Transformer'])
                for i in range(3):
                    for j in range(3):ax.text(j,i,'Not completed' if not np.isfinite(g[i,j]) else f'{g[i,j]:.4f}',ha='center',va='center',color='white' if np.isfinite(g[i,j]) and g[i,j]<.6 else '#172554')
            figure(f'grid_{part}_{metric}',label+': '+chinese,'Each cell equally weights the two Stage A folds at seed 42. Empty cells indicate unavailable results. Lower OFF FPR is better; higher AP and recall are better.',draw_grid,w=9,h=4)
    if (run/'benchmark.json').exists():
        bench=read(run/'benchmark.json')['rows']
        def cost(fig,ax):
            names=[b['network']+'-'+b['representation'] for b in bench]
            ax.barh(names,[b['parameters']/1e6 for b in bench],color='#0284c7');ax.set_xlabel('Model parameters / millions')
        figure('cost','Parameters and front-end cost','Parameter counts come from instantiated models. Throughput, accelerator memory and extrapolations to 40 epochs are recorded in benchmark.json; design estimates are distinguished from measurements.',cost,h=5)
        def tradeoff(fig,ax):
            for b in bench:
                points=[t['summary']['AP'] for t in tasks if t['task']['net']==b['network'] and t['task']['rep']==b['representation'] and t['task']['seed']==42 and not t['task']['id'].endswith(('replay','finetune','pretrain')) and t.get('summary',{}).get('AP') is not None]
                if len(points)==2:
                    x=b.get('MAC_per_endpoint',b['parameters'])/1e6;y=np.mean(points);ax.scatter(x,y,s=50);ax.annotate(b['network']+'-'+b['representation'],(x,y),xytext=(5,5),textcoords='offset points',fontsize=8)
            ax.set_xscale('log');ax.set_xlabel('Model MACs per endpoint / millions (physical preprocessing excluded)');ax.set_ylabel('Selection-partition macro AP; equal fold weights, seed 42')
        if tasks:figure('cost_ap','Recognition performance and compute cost','Model MACs exclude FFT, trajectory rasterization, nonlinear operations and data transfers. Measured execution times are recorded in benchmark.json.',tradeoff)
    if (run/'selection.json').exists():
        selection=read(run/'selection.json');winner=selection['candidate']
        paragraph('Candidate selection and freezing',f"The frozen candidate is {winner['net']}-{winner['rep']}; safeguard status: {'Passed' if winner['protected'] else 'failed; the candidate is diagnostic only'}. Ranking prioritizes safeguards, followed by AP, false positives, parameter count and configuration ID. Stage B, pretraining and final evaluation retain the same candidate.")
        def rankings(fig,ax):
            rows=selection['ranking'];ax.barh([x['net']+'-'+x['rep'] for x in rows],[x['AP'] for x in rows],color=['#059669' if x['protected'] else '#d97706' for x in rows]);ax.invert_yaxis();ax.set_xlabel('Selection-partition macro AP; green: safeguards passed, orange: descriptive only')
        figure('ranking','Candidates and safeguard status','Candidates are ranked by selection-partition AP. Candidates failing recall or false-positive safeguards are marked diagnostic.',rankings)
    if completed:
        decision=read(run/'analysis/decision.json');delta=decision['delta'];ci=np.asarray(decision['CI'])
        def differences(fig,ax):
            val=np.array(list(delta.values()));ax.errorbar(val,np.arange(3),xerr=np.maximum(np.stack((val-ci[0],ci[1]-val)),0),fmt='o',color='#0369a1',capsize=4);ax.axvline(0,color='#64748b',ls='--');ax.set_yticks(range(3),['AP: candidate − reference','Recall: candidate − reference','OFF FPR: candidate − reference']);ax.set_xlabel('Paired 60-second block 95% intervals; higher AP/recall and lower FPR are better')
        figure('paired_ci','Frozen candidate minus N0-W1 under cross-script evaluation','There are 2,000 valid paired replicates; training seeds are not resampled separately. Intervals exclude candidate-selection and threshold-estimation uncertainty.',differences)
        def criteria_plot(fig,ax):
            gg=decision['criteria'];ax.barh([DISPLAY_TEXT.get(key, key) for key in gg],[1]*len(gg),color=['#059669' if v else '#dc2626' for v in gg.values()]);ax.set_xticks([])
            for i,v in enumerate(gg.values()):ax.text(.5,i,'Passed' if v else 'Failed',ha='center',va='center',color='white')
        figure('criteria','Checks against the original frozen criteria','Every threshold must pass to support expanded replication. Failure of one network does not invalidate its entire model family.',criteria_plot)
        paragraph('7. Interpreting the results',f"The cross-script AP difference is {delta['AP']:+.4f}, the recall difference is {delta['Recall']:+.4f}, and the OFF FPR difference is {delta['OFF_FPR']:+.4f}. Seeds with AP improvements in folds A and B number {decision['seed_AP_improvements']['A']}/3, {decision['seed_AP_improvements']['B']}/3. "+'Better selection-partition performance with worse cross-script performance indicates limited transfer. Higher AP with lower recall suggests a shift affecting frozen thresholds or class-conditional distributions. Improvements confined to switching boundaries do not establish a stable operating signature.')
        _detail_figures(run,figure,paragraph)
    paragraph('8. Limitations and further work','The number of records is limited. Cross-script comparisons change appliance settings, background combinations and switching order together, so their individual effects cannot be isolated. Pretraining and aggregate evaluation share physical devices and the measurement chain; cross-brand generalization is untested. Further work requires independent records, devices and background combinations, with continued per-class recall and false-positive evaluation.')
    paragraph('Method sources','ModernTCN (ICLR 2024) and ConvNeXt (2022) inform large kernels and channel interaction; PatchTST (ICLR 2023) informs historical patches; Fryze multi-label research (2020) and three-dimensional V–I research (2024) inform complementary currents and trajectories. SOURCES.md provides links and protocol differences. This experiment uses neither label-triggered segments nor single-appliance classification evaluation.')
    if tasks:
        def histories(fig,ax):
            for a in tasks:
                h=read(run/'tasks'/a['task']['id']/'history.json')
                ax.plot([x['epoch'] for x in h],[x['calibration_balanced_BCE'] for x in h],alpha=.45,lw=.8)
            ax.set_xlabel('Epoch');ax.set_ylabel('Class-balanced BCE on natural calibration endpoints')
        figure('appendix_training','Appendix: training and early-stopping curves','Each line represents an independent fit. Losses from temporary pretraining heads and aggregate tasks are not directly comparable. Full histories are saved per task.',histories)
    for term,meaning in GLOSSARY.items():sections.append(('glossary',term,meaning))
    _html(out/'TDHA_V3_报告.html',sections,font)
    _pdf(out/'TDHA_V3_报告.pdf',sections,font)
    dump(out/'report_manifest.json',{'completed':completed,'files':{p.name:file_hash(p) for p in out.glob('*') if p.is_file() and p.name!='report_manifest.json'},'figures':len(figures),'source_code':code_hash()})
    print('Report written:',out,flush=True)

def _input_figures(run,figure):
    from .physics import physical,harmonics,aligned,raster5
    from .data import load_record
    a=load_record(run/'cache/record2');n=len(a['time']);raw=np.memmap(run/'cache/record2/adc.bin',dtype=np.int16,mode='r',shape=(n,2,1024))
    candidates=np.flatnonzero(a['known'][:,5]&(a['y'][:,5]==1));e=int(candidates[len(candidates)//2]);e=e//5*5+4;block=np.array(raw[e-4:e+1])
    u,i,f,_=physical(block);harm,_=harmonics(block);wave,_=aligned(block);dvi,_,clip=raster5(block)
    def physical_plot(fig,ax):
        ax.remove();axes=fig.subplots(2,3);ms=np.arange(1024)/51.2
        axes[0,0].plot(ms,u[0]);axes[0,0].set_title('Aggregate voltage in one cycle / V')
        axes[0,1].plot(ms,i[0],label='Total current');axes[0,1].plot(ms,f[0],label='Non-active current');axes[0,1].legend(fontsize=8);axes[0,1].set_title('Current / A')
        axes[0,2].plot(ms,wave[0,1]);axes[0,2].set_title('W1 phase-aligned current / A')
        axes[1,0].bar(np.arange(1,26),np.hypot(harm[0,:50:2],harm[0,1:50:2]));axes[1,0].set_title('F absolute harmonic magnitudes')
        axes[1,1].imshow(dvi[0],origin='lower',cmap='viridis');axes[1,1].set_title('DVI total-current density')
        axes[1,2].imshow(dvi[2],origin='lower',cmap='coolwarm');axes[1,2].set_title('DVI signed vertical direction')
        for ax in axes[0]:ax.set_xlabel('Time within cycle / ms')
    figure('input_physics','Physical representations of the same five-cycle block',f'The example uses the median known-ON microwave endpoint, independently of predictions. Trajectory coordinate clipping fraction: {clip:.3%}. It depicts aggregate current, without isolating the microwave waveform.',physical_plot,h=7)
    def coverage(fig,ax):
        counts=[]
        for fold in ('A','B'):
            rows=read(run/'cache'/f'fold{fold}'/'coverage.json');counts.append(np.sum([x['positive'] for x in rows if x['part']=='train'],0))
        arr=np.array(counts);ax.imshow(arr,cmap='Blues',aspect='auto');ax.set_xticks(range(11),NAMES,rotation=35,ha='right');ax.set_yticks([0,1],['Fold A training','Fold B training'])
        for row in range(2):
            for col in range(11):ax.text(col,row,str(arr[row,col]),ha='center',va='center',fontsize=8,color='white' if arr[row,col]>2200 else '#172554')
    figure('class_coverage','Training-positive coverage by output head','Heads with no positive examples contribute no loss and receive no calibrated threshold. The six primary classes have training positives in both folds. Unknown boundaries are excluded separately for each class.',coverage,h=3)

def _detail_figures(run,figure,paragraph):
    import matplotlib.pyplot as plt
    from .runtime import task_id
    from .training import Windows,pairs_for,metadata
    candidate=read(run/'selection.json')['candidate']
    rows=[]
    for model in [('N0','W1'),(candidate['net'],candidate['rep'])]:
        for fold in ('A','B'):
            for seed in (11,42,73):
                path=run/'tasks'/task_id(*model,fold,seed)
                for part in ('time','cross','third'):
                    for row in read(path/f'{part}_metrics.json')['rows']:
                        rows.append(row|{'model':'-'.join(model),'fold':fold,'seed':seed,'part':part})
    dump(run/'report/per_class_metrics.json',rows)
    core=[]
    for part in ('time','cross','third'):
        for model in [('N0','W1'),(candidate['net'],candidate['rep'])]:
            summaries=[read(run/'tasks'/task_id(*model,fold,seed)/f'{part}_metrics.json')['summary'] for fold in ('A','B') for seed in (11,42,73)]
            values=[]
            for key in ('AP','Recall','OFF_FPR','conditional'):
                v=[s[key] for s in summaries if s[key] is not None];values.append(f'{np.mean(v):.4f}' if len(v)==6 else 'Unavailable')
            core.append([{'time':'Within script','cross':'Cross script','third':'Third script'}[part],'-'.join(model),*values])
    def tableplot(fig,ax):
        ax.axis('off');table=ax.table(cellText=core,colLabels=['Partition','Model','Macro AP','Macro recall','Macro OFF FPR','Conditional concordance'],loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,2)
        for (r,c),cell in table.get_celld().items():cell.set_facecolor('#dbeafe' if r==0 else ('#f1f5f9' if r%2 else 'white'));cell.set_edgecolor('#cbd5e1')
    figure('core_table','Core metrics','Metrics equally weight three seeds and two folds. The third script lacks some primary classes, making complete six-class macro AP/recall unavailable. A mean over supported classes does not replace the six-class primary metric. All valid per-class values remain in the CSV and plots below.',tableplot,h=4)
    fields=['model','fold','seed','part','record','class_name','region','present','AP','Recall','Precision','F1','ROC_AUC','OFF_FPR','high_FPR','conditional','coverage']
    with (run/'report/per_class_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    for part,lab in [('time','Within-script later partition'),('cross','Cross script'),('third','Third script combination')]:
        for metric in ('AP','Recall','OFF_FPR'):
            def perclass(fig,ax,part=part,metric=metric):
                for j,model in enumerate(['N0-W1',candidate['net']+'-'+candidate['rep']]):
                    for c in range(11):
                        vals=[x[metric] for x in rows if x['model']==model and x['part']==part and x.get('region')=='main' and x['class']==c and x.get(metric) is not None and (metric!='Recall' or x['present'])]
                        if vals:ax.scatter(np.full(len(vals),c+(j-.5)*.24),vals,s=12,alpha=.35,color=['#0284c7','#e76f51'][j]);ax.plot(c+(j-.5)*.24,np.mean(vals),'_',markersize=15,color=['#0284c7','#e76f51'][j])
                ax.set_xticks(range(11),NAMES,rotation=35,ha='right');ax.set_ylim(-.02,1.02);ax.set_ylabel(metric)
            figure(f'class_{part}_{metric}',lab+': per-class '+metric,'Blue: N0-W1; orange: frozen candidate. Points describe individual records/seeds; horizontal marks show descriptive means. Primary conclusions use hierarchical equal-weight macro metrics, not pooled scatter values. Untrained or unavailable entries are not replaced with zero.',perclass)
    # Choose examples from the baseline seed42 timeline before inspecting candidate-model traces.
    for fold in ('A','B'):
        ds=Windows(run,fold,'P',pairs_for(run,fold,'cross'));meta=metadata(ds);c=5
        roots=[run/'tasks'/task_id(n,r,fold,42) for n,r in [('N0','W1'),(candidate['net'],candidate['rep'])]]
        z=[np.load(p/'cross.npy')[:,c] for p in roots];th=[read(p/'thresholds.json')['logit'][c] for p in roots]
        from sklearn.metrics import precision_recall_curve
        def prplot(fig,ax):
            for j,zz in enumerate(z):
                ok=meta['mask'][:,c].astype(bool);precision,recall,_=precision_recall_curve(meta['y'][ok,c],zz[ok]);ax.plot(recall,precision,label=['N0-W1','Candidate'][j])
            ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.legend();ax.set_xlim(0,1);ax.set_ylim(0,1)
        figure(f'pr_{fold}',f'{fold} fold: microwave precision–recall curve (seed 42)',"Curves pool the fold's endpoints for visual threshold and ranking diagnostics. Primary AP still weights records/classes equally and is not replaced by an area derived from this pooled curve.",prplot)
        def confusion(fig,ax):
            ax.remove();axes=fig.subplots(1,2);ok=meta['mask'][:,c].astype(bool);yy=meta['y'][ok,c]
            for j,axis in enumerate(axes):
                pred=z[j][ok]>=th[j];matrix=np.array([[((yy==0)&~pred).sum(),((yy==0)&pred).sum()],[((yy==1)&~pred).sum(),((yy==1)&pred).sum()]])
                axis.imshow(matrix,cmap='Blues');axis.set_xticks([0,1],['Predicted OFF','Predicted ON']);axis.set_yticks([0,1],['True OFF','True ON']);axis.set_title(['N0-W1','Candidate'][j])
                for row in range(2):
                    for col in range(2):axis.text(col,row,str(matrix[row,col]),ha='center',va='center',color='white' if matrix[row,col]>matrix.max()/2 else 'black')
        figure(f'confusion_{fold}',f'{fold} fold: microwave confusion matrices at frozen thresholds',"Both matrices use the same endpoints and each model's frozen threshold. Seed-averaged probabilities are not treated as the primary model. Matrices for all 11 classes are stored in per-class JSON.",confusion)
        on=(meta['y'][:,c]==1)&meta['mask'][:,c].astype(bool);off=(meta['y'][:,c]==0)&meta['mask'][:,c].astype(bool)
        choices={}
        # Select the overlap-ON case by temporal midpoint of known activity counts, independently of the candidate.
        ii=np.flatnonzero(on)
        if len(ii):
            groups=np.split(ii,np.flatnonzero((np.diff(ii)>1)|(np.diff(meta['record'][ii])!=0))+1);groups.sort(key=len);g=groups[len(groups)//2];choices['典型ON']=int(g[len(g)//2]);choices['低功率ON']=int(ii[np.argmin(meta['power'][ii])])
        ii=np.flatnonzero(off)
        if len(ii):choices['高功率OFF']=int(ii[np.argmax(meta['power'][ii])])
        for name,m,fn in [('FP',off&(z[0]>=th[0]),np.argmax),('FN',on&(z[0]<th[0]),np.argmin)]:
            ii=np.flatnonzero(m)
            if len(ii):choices[name]=int(ii[fn(z[0][ii])])
            else:paragraph(f'{fold} fold {name} example','The seed-42 reference has no example of this type in this fold.')
        for name,idx in choices.items():
            m=(meta['record']==meta['record'][idx])&(np.abs(meta['time']-meta['time'][idx])<=30000000);sec=(meta['time'][m]-meta['time'][idx])/1e6
            def case(fig,ax,m=m,sec=sec):
                for j,zz in enumerate(z):ax.plot(sec,1/(1+np.exp(-np.clip(zz[m],-50,50))),label=['Reference N0-W1','Frozen candidate'][j],lw=1.3);ax.axhline(1/(1+np.exp(-th[j])),ls='--',alpha=.5,color=['#0284c7','#e76f51'][j])
                ax.fill_between(sec,0,1,where=meta['y'][m,c]==1,color='#059669',alpha=.12,label='True ON');ax.fill_between(sec,0,1,where=~meta['mask'][m,c].astype(bool),color='#64748b',alpha=.3,label='Unknown label');ax.legend(loc='upper right',fontsize=8);ax.set_xlabel('Time from reference-selected endpoint / seconds');ax.set_ylim(0,1)
            figure(f'case_{fold}_{name}',f'{fold} fold: microwave {DISPLAY_TEXT.get(name, name)}','Dashed lines show each frozen threshold. Both models share the same segment and unknown-label shading. Examples aid interpretation and do not replace metrics over the natural distribution.',case)
    paragraph('Scope of the pretraining comparison','The following supplemental comparison contrasts training from scratch with pretrained fine-tuning at seed 42. The temporal backbone is reinitialized, and normalization is fitted on the aggregate training partition. The primary comparison retains the frozen candidate across two folds and three seeds.')
    vals=[]
    for fold in ('A','B'):
        for suffix in ('','-finetune'):
            p=run/'tasks'/task_id(candidate['net'],candidate['rep'],fold,42,suffix)/'cross_metrics.json'
            if p.exists():vals.append((fold+(' from scratch' if not suffix else ' pretrained'),read(p)['summary']))
    def preplot(fig,ax):
        x=np.arange(len(vals));ax.bar(x,[a['AP'] for _,a in vals],color='#0d9488');ax.set_xticks(x,[n for n,_ in vals]);ax.set_ylabel('Cross-script macro AP')
    figure('pretrain','Supplemental single-appliance representation transfer','This comparison shares physical devices and the measurement chain. It does not demonstrate transfer to unseen devices or brands.',preplot)

def _html(path,sections,font):
    pieces=[];gloss=[]
    for kind,title,content in sections:
        if kind=='figure':
            image,caption=content;uri=base64.b64encode(image.read_bytes()).decode();pieces.append(f'<section><h2>{html.escape(title)}</h2><img alt="{html.escape(title)}" src="data:image/png;base64,{uri}"><p>{html.escape(caption)}</p></section>')
        elif kind=='glossary':gloss.append(f'<article class="term"><h3>{html.escape(title)}</h3><p>{html.escape(content)}</p></article>')
        else:pieces.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(content)}</p></section>')
    css='body{margin:0;background:#eef3f8;color:#17324d;font-family:TDHA,sans-serif;line-height:1.85}main{max-width:1120px;margin:auto;padding:36px}section,article{background:white;border:1px solid #dce5ee;border-radius:14px;padding:25px;margin:18px 0}h1{font-size:32px}h2{font-size:23px}p{font-size:16px}img{width:100%;height:auto}input{font:inherit;padding:12px;width:90%;border:1px solid #94a3b8;border-radius:8px}.term[hidden]{display:none}footer{padding:30px;color:#64748b}'
    data=base64.b64encode(font.read_bytes()).decode()
    document='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TDHA V3 high-frequency multi-label experiment report</title><style>@font-face{font-family:TDHA;src:url(data:font/ttf;base64,'+data+')}'+css+'</style><main><h1>TDHA V3: appliance supply states from aggregate measurements</h1><p>High-frequency representations × network architectures | Offline report | Exploratory evidence</p>'+''.join(pieces)+'<section><h2>Appendix: searchable glossary</h2><input id="search" type="search" placeholder="Search: AP, V-I, thresholds, GRU..."><div id="terms">'+''.join(gloss)+'</div></section><footer>Images, fonts, styles and glossary search are embedded for offline reading.</footer></main><script>document.getElementById("search").addEventListener("input",function(){const q=this.value.trim().toLowerCase();document.querySelectorAll(".term").forEach(e=>e.hidden=!e.textContent.toLowerCase().includes(q));});</script></html>'
    path.write_text(document,encoding='utf-8')

def _pdf(path,sections,font):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,KeepTogether
    from reportlab.lib.colors import HexColor
    pdfmetrics.registerFont(TTFont('TDHA',str(font)))
    body=ParagraphStyle('cn',fontName='TDHA',fontSize=10,leading=17,wordWrap='CJK',textColor=HexColor('#17324d'))
    head=ParagraphStyle('head',parent=body,fontSize=15,leading=22,spaceBefore=15,spaceAfter=8)
    flow=[Paragraph('TDHA V3: appliance supply states from aggregate measurements',head)]
    for kind,title,content in sections:
        flow.append(Paragraph(html.escape(title),head))
        if kind=='figure':
            image,caption=content
            from PIL import Image as PILImage
            with PILImage.open(image) as im:w,h=im.size
            flow.extend([Image(str(image),width=490,height=490*h/w),Paragraph(html.escape(caption),body),Spacer(1,8)])
        else:flow.append(Paragraph(html.escape(content),body))
    def page(canvas,doc):
        canvas.setFont('TDHA',8);canvas.drawString(45,24,'TDHA controlled aggregate recordings · Exploratory evidence');canvas.drawRightString(550,24,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=(595,842),leftMargin=45,rightMargin=45,topMargin=40,bottomMargin=45).build(flow,onFirstPage=page,onLaterPages=page)

def pack(run,models=False):
    run=Path(run);dest=run/'downloads';dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest/'TDHA_V3_报告与汇总.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((run/'report').rglob('*')):
            if p.is_file():z.write(p,p.relative_to(run))
        for name in ('selection.json','freeze.json','reproducibility.json','execution_complete.json','blocker.json','benchmark.json','operator_preflight.json','analysis/decision.json','analysis/bootstrap.json','analysis/absent.json','audit/exposure.json','audit/dependencies.json','audit/environment.json'):
            p=run/name
            if p.exists():z.write(p,name)
        # Include existing task metrics even when no optional report was rendered.
        for p in sorted((run/'tasks').glob('*/*')):
            if p.is_file() and (p.name in ('memory.json','done.json','readout.json','thresholds.json') or p.name.endswith(('_metrics.json','_metrics.csv'))):
                z.write(p,p.relative_to(run))
        for folder in ('audit/errors','audit/operator_preflight','audit/operator_preflight_skips','audit/pretrain_labels','audit/resume_upgrade'):
            for p in sorted((run/folder).rglob('*')):
                if p.is_file():z.write(p,p.relative_to(run))
    if models:
        with zipfile.ZipFile(dest/'TDHA_V3_模型与预测.zip','w',zipfile.ZIP_STORED) as z:
            for p in sorted((run/'tasks').glob('*/*')):
                if p.name in ('best.pt','thresholds.json','done.json','readout.json') or p.suffix=='.npy':z.write(p,p.relative_to(run))
    print('Result archives:',dest,flush=True)
