"""Selected input contracts and fail-closed loading; no training or data reads."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from wattsign import selected_models as selected

class SelectedModelTests(unittest.TestCase):
    def test_hash_mismatch_blocks_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"untrusted.joblib"
            p.write_bytes(b"not a model")
            with patch("builtins.__import__",wraps=__import__) as imports:
                with self.assertRaisesRegex(ValueError,"SHA-256"):
                    selected.load_model("plaid-resistive",p,expected_sha256="0"*64)
            self.assertFalse(any(c.args[0] in ("joblib","torch") for c in imports.call_args_list))

    def test_10s_normalization_masks_and_past_window(self):
        x=np.zeros((70,18),np.float32)
        x[:,0]=np.arange(70);x[:,3]=1
        x[:,6:12]=9
        norm={"means":[10.0]*18,"stds":[2.0]*18}
        before=x.copy()
        out=selected.ukdale_10s_windows(x,np.array([63]),norm)
        self.assertEqual(out.shape,(1,18,64))
        np.testing.assert_array_equal(out[0,0],(np.arange(64)-10)/2)
        self.assertTrue((out[:,6:12]==0).all())
        self.assertTrue((out[:,1:3]==0).all())
        np.testing.assert_array_equal(x,before)
        x[64:]=12345
        np.testing.assert_array_equal(out,selected.ukdale_10s_windows(x,np.array([63]),norm))
        with self.assertRaises(ValueError):selected.ukdale_10s_windows(x,np.array([62]),norm)

    def test_1s_right_boundary_and_optional_channels(self):
        x=np.zeros((650,18),np.float32);x[:,0]=np.arange(650);x[:,3]=1
        out=selected.ukdale_1s_windows(x,np.array([63]),{"0":{"mean":0,"std":1}})
        self.assertEqual(out.shape,(1,42,640))
        np.testing.assert_array_equal(out[0,0],np.arange(640))
        self.assertTrue((out[:,18:]==0).all())
        x[640:]=888
        np.testing.assert_array_equal(out,selected.ukdale_1s_windows(x,np.array([63]),{"0":{"mean":0,"std":1}}))

    def test_waveform_masks_do_not_populate_feature_values(self):
        base=np.ones((2560,36),np.float32);mask=np.ones(2560,np.uint8);mask[10]=0
        out=selected.tdha_waveform_windows(base,mask,np.array([2559]))
        self.assertEqual(out.shape,(1,520,2560))
        self.assertTrue((out[:,:36]==1).all())
        self.assertTrue((out[:,36:278]==0).all())
        np.testing.assert_array_equal(out[0,278],mask)
        self.assertTrue((out[:,294:]==0).all())

    def test_shared_asset_subdirectory_is_found_without_claiming_verified(self):
        from load_signature_analysis.notebook_view import experiment_markdown
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"wattsign-models/results/ukdale-1s").mkdir(parents=True)
            case={"id":"ukdale-1s","title":"Saved case","status":"recorded","assets":[{"filename":"wattsign-models.zip","content_path":"results/ukdale-1s","url":None}]}
            with patch.dict(os.environ,{"NILM_RESULTS_ROOT":str(root)}):text=experiment_markdown(case)
            self.assertIn("Extracted files available locally",text)
            self.assertNotIn("verified",text.lower())
            self.assertIn("mailto:ruizhang.eric@gmail.com",text)
            self.assertNotIn("](https://",text)
            case["assets"][0]["content_path"]="../outside"
            with self.assertRaises(ValueError):experiment_markdown(case)

    def test_saved_case_view_reads_all_five_summaries_without_models(self):
        from load_signature_analysis.notebook_view import show_experiment
        root=Path(__file__).resolve().parents[1]
        cases=(*selected.CASES,"tdha-npu")
        with patch("IPython.display.display") as display, patch.object(selected,"load_model",side_effect=AssertionError("display must not load models")):
            for case in cases:
                show_experiment(case,root)
        self.assertEqual(display.call_count,10)
        for i in range(0,10,2):
            markdown=display.call_args_list[i].args[0].data
            self.assertIn("### Results",markdown)
            self.assertNotRegex(markdown, r"[\u3400-\u9fff]")
            self.assertIn("| AP",markdown)
            self.assertNotIn("summary.json/README",markdown)
