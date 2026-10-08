"""Synthetic I/O contract smoke test; not a substitute for running GitHub's full real CSVs."""
from __future__ import annotations
from pathlib import Path
import hashlib
import os
import json
import shutil
import subprocess
import sys
import tempfile
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parents[1]
SOURCE_08 = Path(os.environ.get('KAMP_TEST_SOURCE_08', str(HERE.parent/'08_Modeling')))
NAMES = {'08_01_model_comparison.py': '08_01_model_comparison(1).py',
         '08_02_Model_Error_Analysis.py': '08_02_Model_Error_Analysis.py',
         '08_03_Final_Alarm_Policy.py': '08_03_Final_Alarm_Policy.py'}


def csv(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)


def run():
    with tempfile.TemporaryDirectory(prefix='KAMP_11_12_test_') as d:
        root=Path(d)
        model=root/'08_Modeling'
        model.mkdir()
        for n, old in NAMES.items():
            f = SOURCE_08/old
            if not f.exists() and (SOURCE_08/n).exists():
                f = SOURCE_08/n
            shutil.copy2(f, model/n)
        before={n:hashlib.sha256((model/n).read_bytes()).hexdigest() for n in NAMES}
        status=[]
        for source, seg, label in [('normal','N0000',0), ('abnormal','A0001',1)]:
            for i in range(6):
                status.append(dict(source=source,segment_id=seg,source_row=i,pos_in_seg=9+i,seg_len=30,
                   label=label, eval_set=('TEST' if i<2 else ('OOF' if i<5 else np.nan)),
                   eval_status='not_evaluable' if i==5 else 'evaluated',
                   final_alarm=(1 if source=='abnormal' and i<2 else (1 if i==0 else 0)),
                   alarm_level=('경보' if (source=='abnormal' and i<2) or i==0 else '정상')))
        for row in status:
            if row['eval_status']=='not_evaluable':
                row['final_alarm']=np.nan
                row['alarm_level']=np.nan
        # TEST 4 and OOF 6, one unassessed for each source
        test=[r for r in status if r['eval_set']=='TEST']
        for r in test:
            r['A_alarm']=int(r['final_alarm'])
            r['B_alarm']=int(r['final_alarm'])
            r['A_prob']=.9 if r['final_alarm'] else .1
            r['B_normal_pct']=99 if r['final_alarm'] else 10
        results=model/'results'
        csv(results/'final/test_predictions_final.csv',test)
        cfg={'A':{'model':'RandomForest','features':'F4_+H1잔차','threshold':.3},
             'B':{'model':'Mahalanobis','features':'F4_+H1잔차','threshold':140},
             'k':1,'final_rule':'A·B 모두 이상이면 즉시 경보'}
        (results/'final/final_config.json').write_text(json.dumps(cfg,ensure_ascii=False))
        csv(results/'final/rows_final_status.csv',status)
        oo=[r for r in status if r['eval_set']=='OOF']
        csv(results/'oof_predictions_A.csv',[{**r,'alarm':int(r['final_alarm'])} for r in oo])
        csv(results/'oof_predictions_B.csv',[{**r,'alarm':int(r['final_alarm'])} for r in oo])
        csv(results/'final/importance_family.csv',[dict(group='전류 크기',PR_AUC_drop_mean=.08)])
        csv(results/'final/importance_feature.csv',[dict(group='cur_crest',PR_AUC_drop_mean=.006)])
        csv(results/'final/ablation_table.csv',[dict(framework='A',model='RMS-IQR',features='F0_RMS',CV_PR_AUC=.55,TEST_F1=.76,TEST_Recall=.70,TEST_FalseAlarm=.006)])
        csv(results/'final/policy_comparison.csv',
          [dict(eval_set=e,rule='A·B 모두(경보)',k=k,Recall=1.0 if k==1 else .9,F1=.98 if k==1 else .90,
                FalseAlarmRate=.002 if k==1 else .001, AbnSegDetectRate=1.0,NormSegFalseAlarmRate=.03)
           for e in ['OOF','TEST'] for k in [1,2]])
        csv(results/'Error_analysis/C_Comparison/common_reason_thresholds.csv',
            [{'metric':k,'value':v} for k,v in dict(H1_AI0_Q99=.05,H1_AI1_Q99=.05,H2_Q01=-.1,H2_Q99=.1,CURRENT_Q01=50,CURRENT_Q99=200).items()])
        csv(results/'Error_analysis/C_Comparison/common_cycle_context_by_segment.csv',
            [dict(source='abnormal',segment_id='A0001',cycle_context_available=True,cycle_timing_break=True,cycle_shape_break=True,
                  cycle_current_break=False,cycle_response_break=True)])
        raw=[]; man=[]
        for r in status:
            raw.append({k:r[k] for k in ['source','segment_id','source_row','seg_len']} | {'elapsed_sec':float(3650 if r['source']=='normal' else 10)})
            man.append({k:r[k] for k in ['source','segment_id','source_row']} |
                       dict(RMS_Detected=bool(r['source']=='abnormal'),H1_Residual_AI0=.08 if r['source']=='abnormal' else .01,
                            H1_Residual_AI1=.08 if r['source']=='abnormal' else .01,H2_ND=.02,H3_Current_Amplitude=99))
        csv(root/'data/processed/preprocessed_data.csv',raw)
        csv(root/'data/10_04_manifest_all.csv',man)
        output=root/'11_12_results'
        q=subprocess.run([sys.executable,str(HERE/'run_11_12.py'),'--root',str(root),'--output',str(output)],capture_output=True,text=True)
        print(q.stdout)
        if q.returncode:
            print(q.stderr)
            raise AssertionError(f'pipeline failed: {q.returncode}')
        after={n:hashlib.sha256((model/n).read_bytes()).hexdigest() for n in NAMES}
        assert before==after, '08 MODEL MODIFIED'
        c=pd.read_csv(output/'11/11_condition_performance.csv')
        assert set(c.eval_set)=={'OOF','TEST'}
        assert ((c.condition=='ALL')&(c.eval_set=='TEST')).sum()==1
        decision=pd.read_csv(output/'12/12_operational_decision_all_rows.csv')
        assert len(decision)==len(status)
        assert decision.loc[decision.eval_status=='not_evaluable','operational_state'].eq('평가불가').all()
        assert decision.operational_state.eq('경보').sum()==3, 'confirmed 08 alarm lost during operator state mapping'
        assert not ({'label','source','segment_id'} & set(decision.columns)), 'ground truth leaked into operator decision queue'
        assert len(pd.read_csv(output/'12/12_policy_tradeoff_READONLY.csv'))==4
        print('TEST OK: schema, condition FP/FN, decision states, no label leakage, model hashes unchanged')
        # 지정 커밋 08 모델이 달라지면 실행을 거부해야 한다.
        (model/'08_01_model_comparison.py').write_bytes((model/'08_01_model_comparison.py').read_bytes() + b'\n#changed')
        guarded=subprocess.run([sys.executable,str(HERE/'run_11_12.py'),'--root',str(root),'--output',str(output)],capture_output=True,text=True)
        assert guarded.returncode != 0 and '08' in (guarded.stdout+guarded.stderr)
        print('TEST OK: fail-closed guard on modified 08 source')

if __name__=='__main__':
    run()
