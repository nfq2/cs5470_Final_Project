"""Synthetic baseline checks; never access the Perlmutter dataset."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from PIL import Image
import train
from model import WireModel
from common import protocol

class TinyEncoder(nn.Module):
    num_features = 3
    pretrained_cfg = {'mean': protocol.IMAGENET_MEAN, 'std': protocol.IMAGENET_STD}
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
    def forward(self, x):
        return self.bn(x).mean((2, 3))

class ProtocolTests(unittest.TestCase):
    def test_baseline_training_and_selection(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (8,8), (30,60,90)).save(root/'x.png')
            cases = [dict(paths=['x.png']*7, episode=i, label=i%2,
                image_roles=['goal_map','history_1','history_2','history_3','history_4','history_5','now'],
                history_offsets_s=[-5,-4,-3,-2,-1,0], depth_relative_mm=[-i]*5+[0]) for i in range(6)]
            (root/'cases.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in cases))
            (root/'splits.json').write_text(json.dumps([dict(fold=0,train=[0,1],validation=[2,3],test=[4,5])]))
            encoder = TinyEncoder()
            before = {k:v.clone() for k,v in encoder.state_dict().items()}
            cpu = torch.device('cpu')
            with patch.object(sys,'argv',['train.py','--data-root',str(root),'--output-dir',str(root/'run'),'--num-workers','0']), patch('train.timm.create_model',return_value=encoder), patch('train.torch.cuda.is_available',return_value=True), patch('train.torch.device',return_value=cpu), patch('builtins.print'):
                train.main()
            saved = torch.load(root/'run/best.pt',weights_only=True)
            self.assertEqual(saved['config']['protocol'],protocol.NAME)
            self.assertEqual(saved['config']['optimizer'],'Adam')
            self.assertEqual(saved['config']['weight_decay'],0)
            self.assertEqual(saved['preprocessing']['size'],[448,448])
            history = [json.loads(l) for l in (root/'run/history.jsonl').read_text().splitlines()]
            self.assertEqual(len(history),20)
            self.assertEqual(saved['epoch'],min(history,key=lambda r:r['validation']['loss'])['epoch'])
            for key,value in encoder.state_dict().items():
                torch.testing.assert_close(value,before[key],rtol=0,atol=0)
            torch.testing.assert_close(saved['model']['history_mean'][:5],torch.full((5,),-.5))

    def test_custom_settings_rejected_without_experiment_flag(self):
        for flag in ('--finetune','--random-init','--dropout=0.1'):
            with patch.object(sys,'argv',['train.py','--data-root','unused','--output-dir','unused',flag]), self.assertRaises(SystemExit):
                train.main()

    def test_depth_first_history(self):
        with patch('model.timm.create_model',return_value=TinyEncoder()):
            model = WireModel(pretrained=False,history_order='depth_then_time')
            model.fit_history_normalization([dict(depth_relative_mm=[-2]*5+[0],history_offsets_s=[-5,-4,-3,-2,-1,0])])
            torch.testing.assert_close(model.history_mean[:6],torch.tensor([-2.,-2,-2,-2,-2,0]))
            torch.testing.assert_close(model.history_mean[6:],torch.tensor([-5.,-4,-3,-2,-1,0]))
