import argparse
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from .common import *

def reference_build(data,legacy,out):
    data=Path(data);legacy=Path(legacy);sources={}
    for p in sorted(data.iterdir()):
        if p.name=='Background.zip' or p.suffix.lower() not in ('.zip','.xlsx'):continue
        print('Checking source',p.name,flush=True);before=p.stat();h=file_hash(p);after=p.stat()
        require(before.st_mtime_ns==after.st_mtime_ns and before.st_size==after.st_size,'Source file changed during hashing')
        sources[p.name]={'sha256':h,'size':after.st_size}
    regression={}
    for r in range(1,8):
        with np.load(legacy/f'agg{r}_timeline.npz') as z:
            regression[str(r)]={k:array_hash(z[k]) for k in ('time','index','basic','bmask','continuity')}
            regression[str(r)].update(microwave_y=array_hash(z['y']),microwave_known=array_hash(z['known']))
    dump(out,{'sources':sources,'regression':regression,'来源':'V2 继承的 V1 逐周期时间线；仅打包哈希，不打包原始数据','scope':'TDHA only'})

def main():
    p=argparse.ArgumentParser(description='TDHA V3: fixed exploratory multi-label queue using at most three NPUs')
    p.add_argument('command',choices=['doctor','preflight','prepare','benchmark','run','report','pack','worker','reference','test'])
    p.add_argument('--data',default=os.environ.get('TDHA_DATA','datasets/TDHA/V2'))
    p.add_argument('--run',default=os.environ.get('TDHA_RUN',str(ROOT/'outputs/pilot_v3')))
    p.add_argument('--devices',default='0,1,2');p.add_argument('--reference',default=str(ROOT/'references/source_manifest.json'))
    p.add_argument('--legacy');p.add_argument('--task');p.add_argument('--action',choices=['fit','readout'],default='fit')
    p.add_argument('--resume',action='store_true',help='Resume from the latest complete optimization step of the same task')
    p.add_argument('--font',help='Optional report export: explicit licensed CJK TTF path')
    p.add_argument('--models',action='store_true',help='Include a separate model/prediction archive when packing')
    p.add_argument('--skip-preflight',action='store_true',help='preflight only: record a skipped NPU operator check; keep other checks')
    args=p.parse_args();run=Path(args.run)
    if args.skip_preflight and args.command!='preflight':p.error('--skip-preflight applies only to preflight; use bash start.sh --skip-preflight for the full queue')
    if args.command in ('preflight','benchmark','run','worker'):
        from .session import require_tmux
        try:require_tmux()
        except RuntimeError as e:p.error(str(e))
    font=None
    if args.command in ('doctor','run','report'):
        from .reporting import resolve_report_font,validate_report_font
        try:
            font=resolve_report_font(args.font,required=args.command=='report')
            if font is not None:validate_report_font(font)
        except Exception as exc:
            p.error(str(exc))
    error_command=args.command
    try:
        if args.command=='reference':reference_build(args.data,args.legacy,args.reference)
        elif args.command=='test':
            import unittest
            result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
            dump(ROOT/'local_validation/test_summary.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'python':sys.version,'code':code_hash(),'scope':'CPU单元测试与合成恢复测试；不等于NPU验收'})
            require(result.wasSuccessful(),'Local tests failed')
        elif args.command=='doctor':
            from .runtime import doctor
            doctor(run,include_report=font is not None)
        elif args.command=='prepare':
            from .data import prepare
            prepare(args.data,run,args.reference)
        elif args.command=='preflight':
            from .npu_preflight import preflight,record_skip
            (record_skip if args.skip_preflight else preflight)(run,[int(x) for x in args.devices.split(',')])
        elif args.command=='benchmark':
            cards=[int(x) for x in args.devices.split(',')];require(len(cards)<=3,'Specify at most three NPUs')
            from .runtime import available,benchmark
            require(available(cards[0]),'The preflight NPU is occupied by another task')
            os.environ['ASCEND_RT_VISIBLE_DEVICES']=str(cards[0])
            from .training import backend
            benchmark(run,backend())
        elif args.command=='run':
            from .runtime import run_queue
            from .reporting import report,pack
            if (run/'tasks').exists() and not args.resume:raise Blocked('Run already exists; use --resume for the same run or choose a new output directory')
            run_queue(args.data,run,[int(x) for x in args.devices.split(',')],args.reference)
            primary_error=None
            try:
                if (run/'execution_complete.json').exists() and not (run/'analysis/decision.json').exists():
                    error_command='analysis'
                    from .analysis import analyze
                    analyze(run)
                if font is not None:
                    error_command='report'
                    report(run,font)
            except Exception as exc:
                primary_error=exc
                raise
            finally:
                # Keep the original analysis/report error if packing also fails.
                try:pack(run,args.models)
                except Exception as pack_error:
                    if primary_error is None:
                        error_command='pack'
                        raise
                    dump(run/'audit/errors'/f'pack-{time.time_ns()}.json',{'command':'pack','type':type(pack_error).__name__,'reason':str(pack_error),'traceback':traceback.format_exc()})
                    traceback.print_exc()
        elif args.command=='worker':
            from .training import backend,fit,readout
            device=backend();task=read(args.task)
            (fit if args.action=='fit' else readout)(run,task,device)
        elif args.command=='report':
            from .reporting import report
            report(run,font)
        elif args.command=='pack':
            from .reporting import pack
            pack(run,args.models)
    except Exception as e:
        traceback.print_exc()
        if args.command not in ('reference','test'):
            error={'command':error_command,'type':type(e).__name__,'reason':str(e),'traceback':traceback.format_exc(),'说明':'停止不是实验阴性结论；不改变模型或删减数据来凑齐结果'}
            dump(run/'audit/errors'/f'{error_command}-{time.time_ns()}.json',error)
            # Reporting/packing errors must not replace the underlying training error.
            if error_command not in ('report','pack'):
                dump(run/'blocker.json',error)
            elif not (run/'blocker.json').exists() and not (run/'execution_complete.json').exists():
                dump(run/'blocker.json',error)
        return 1
    return 0
