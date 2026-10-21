"""Selection and continuation tests; scheduler calls are mocked."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import manage

class StageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp');self.root=Path(self.temp.name)
        (self.root/'control').mkdir();self.original=manage.ROOT;manage.ROOT=self.root
        self.cohorts={'piv_validation':{'case_ids':['v'+str(i) for i in range(24)]},
                      'piv_terminal':{'case_ids':['t'+str(i) for i in range(96)]}}
        (self.root/'control/cohorts.json').write_text(json.dumps(self.cohorts))
    def tearDown(self):manage.ROOT=self.original;self.temp.cleanup()
    def jobs(self,stage,status='completed'):
        output=self.root/stage;output.mkdir()
        (output/'checkpoint.json').write_text(json.dumps({'status':status}))
        jobs=[{'name':'job','method':'mac','task':'piv','output':str(output)}]
        (self.root/'control'/f'mac_piv_{stage}_jobs.json').write_text(json.dumps(jobs))
        return jobs
    def records(self,n):
        return [{'method':p,'case_id':str(i),'status':'ok','error':str(e),'solve_wall_seconds':'20'}
                for p,e in [('low',.2),('high',.1)] for i in range(n)]
    def test_calibration_excludes_incomplete_profile(self):
        self.jobs('calibration');rows=self.records(4);rows[-1]['status']='timeout';rows[-1]['error']='nan'
        with patch.object(manage,'rows',return_value=rows),patch.object(manage,'schedule') as schedule:
            manage.advance('mac_piv','calibration',1)
        family,stage,jobs=schedule.call_args.args
        self.assertEqual(stage,'validation');self.assertEqual(len(jobs),2)
        self.assertTrue(all(j['profiles']==['low'] for j in jobs))
        self.assertEqual(sum((j['case_ids'] for j in jobs),[]),self.cohorts['piv_validation']['case_ids'])
    def test_validation_freezes_winner_before_test(self):
        self.jobs('validation')
        with patch.object(manage,'rows',return_value=self.records(24)),patch.object(manage,'schedule') as schedule:
            manage.advance('mac_piv','validation',1)
        family,stage,jobs=schedule.call_args.args
        self.assertEqual(stage,'terminal');self.assertEqual(len(jobs),8)
        self.assertTrue(all(j['profiles']==['high'] for j in jobs))
        frozen=json.loads((self.root/'control/mac_piv_selected.json').read_text())
        self.assertEqual(frozen['status'],'frozen_before_test')
    def test_all_failed_stops(self):
        self.jobs('calibration');rows=self.records(4)
        for row in rows:row['status']='failed'
        with patch.object(manage,'rows',return_value=rows),patch.object(manage,'schedule') as schedule:
            with self.assertRaisesRegex(RuntimeError,'No profile'):manage.advance('mac_piv','calibration',1)
        schedule.assert_not_called()
    def test_one_resume_then_stop(self):
        jobs=self.jobs('terminal','paused')
        with patch.object(manage,'schedule') as schedule:manage.advance('mac_piv','terminal',1)
        self.assertEqual(schedule.call_args.args,('mac_piv','terminal',jobs,2,[0]))
        with self.assertRaisesRegex(RuntimeError,'stopped'):manage.advance('mac_piv','terminal',2)
    def test_duplicate_result_rejected(self):
        rows=self.records(4)
        with self.assertRaisesRegex(ValueError,'duplicate'):manage.statistics(rows+rows[:1])
if __name__=='__main__':unittest.main()
