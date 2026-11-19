"""Checks for capture immutability and geometric parity measurements."""
import unittest
import sys
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch

from capture_soma import TensorCapture
from compare_soma import compare, compare_pose_decisions, duration_decision
from compare_soma_layers import valid_sequence_prefix, compare as compare_fp32_layers, PREFIXES
from compare_soma_mixed_layers import compare as compare_mixed_layers, temporal_prefix
from check_soma_chain import evaluate
from check_soma_api_chain import compare as compare_api_chain
from export_soma_demo import build_plan
from audit_soma_matmul import aligned_block_product, split_half_product


def motion():
    features = np.zeros((1, 64, 366), dtype=np.float32)
    features[:, :, 92:272] = np.tile([1, 0, 0, 0, 1, 0], 30)
    return {'output/pred_global_motions': features,
            'output/pred_num_tokens': np.array([8], dtype=np.int32)}


class SOMAToolTests(unittest.TestCase):
    def test_semantic_coverage_excludes_weights_but_not_missing_layers(self):
        class Capture(dict):
            @property
            def files(self): return list(self)
        reference = Capture({'output/pred_num_tokens': np.array([16]),
                             'graph//mmm_net_1/Add_2_output_0': np.zeros((2, 3))})
        native = {'tap.0': np.zeros((2, 3)), 'tap.1': np.zeros((2, 3))}
        metadata = {'tap.0.name': 'soma/0/layers.0.self_attn.in_proj_weight',
                    'tap.0.operation': 'CPY', 'tap.1.name': 'soma/1/_positioned_input',
                    'tap.1.operation': 'ADD'}
        result = compare_mixed_layers(reference, metadata, native)
        self.assertTrue(result['semantic_coverage_complete'])
        self.assertEqual(result['semantic_taps'], 1)
        self.assertEqual(result['excluded_weight_conversions'], 1)
        native['tap.2'] = np.zeros((2, 3))
        metadata.update({'tap.2.name': 'soma/2/missing_layer', 'tap.2.operation': 'ADD'})
        self.assertFalse(compare_mixed_layers(reference, metadata, native)['semantic_coverage_complete'])

    def test_public_api_chain_rejects_duration_and_contact_failures(self):
        poses = np.zeros((18, 2, 7), dtype=np.float32)
        poses[..., 6] = 1
        expected = dict(tokens=np.array([6]), corrected=poses, world=poses,
                        contacts=np.zeros((18, 4), dtype=np.float32))
        actual = {key: np.array(value, copy=True) for key, value in expected.items()}
        actual['tokens'] = 6
        limits = dict(position_m=1e-4, rotation_deg=.01, contacts_absolute=1e-4)
        self.assertTrue(compare_api_chain(actual, expected, limits)['passed'])
        actual['tokens'] = 7
        self.assertFalse(compare_api_chain(actual, expected, limits)['passed'])
        actual['tokens'] = 6
        actual['contacts'][0, 0] = .001
        self.assertFalse(compare_api_chain(actual, expected, limits)['passed'])
        actual['contacts'][0, 0] = np.nan
        self.assertFalse(compare_api_chain(actual, expected, limits)['passed'])
        actual['contacts'][0, 0] = 0
        actual['world'] = actual['world'][:-1]
        self.assertFalse(compare_api_chain(actual, expected, limits)['passed'])

    def test_full_fp32_trace_keeps_root_and_pose_projections_separate(self):
        class Capture(dict):
            @property
            def files(self): return list(self)
        reference=Capture();metadata={};native={}
        for i,(component,index,name,value) in enumerate([
                ('root',9,'_proj_local_pose',1),
                ('pose',0,'_proj_local_pose',2),
                ('decoder',0,'decoder.model.0',3)]):
            key=f'tap.{i}'
            metadata[key+'.name']=f'soma/{index}/{name}'
            metadata[key+'.operation']='ADD'
            native[key]=np.full((2,3),value,dtype=np.float32)
            reference['layer/'+PREFIXES[component]+name+'/0/output']=native[key].copy()
        with patch('compare_soma_layers.read_gguf',return_value=(metadata,native)):
            report=compare_fp32_layers(reference,'unused')
        self.assertEqual(report['mapped_layers'],3)
        self.assertTrue(all(row['max_absolute']==0 for row in report['layers']))

    def test_alignment_remainder_precedes_full_blocks(self):
        a=np.zeros((1,20),np.float16);b=np.ones((20,1),np.float16)
        a[0,:4]=2**-13;b[:4,0]=2**-13
        a[0,4]=1;a[0,5]=-1
        self.assertEqual(float(aligned_block_product(a,b,16,25)[0,0]),0)
        self.assertEqual(float(aligned_block_product(a,b,16,25,residue_first=True)[0,0]),2**-24)

    def test_split_probe_rounds_partials_before_fp32_reduction(self):
        a=np.zeros((1,32),np.float16);b=np.ones((32,1),np.float16)
        a[0,0]=1;a[0,1]=2**-11;a[0,16]=-1
        self.assertEqual(float(aligned_block_product(a,b,16,25)[0,0]),2**-11)
        self.assertEqual(float(split_half_product(a,b,16)[0,0]),0)
        with self.assertRaises(ValueError):split_half_product(a,b,15)

    def test_alignment_probe_truncates_before_cancellation(self):
        a=np.array([[1.5,2**-13,2**-13,2**-13,2**-13,-1.5]],np.float16)
        b=np.array([[1.5],[2**-13],[2**-13],[2**-13],[2**-13],[1.5]],np.float16)
        self.assertEqual(float((a.astype(np.float64)@b.astype(np.float64))[0,0]),2**-24)
        self.assertEqual(float(aligned_block_product(a,b,8,25)[0,0]),0)
        self.assertEqual(float(aligned_block_product(a,b,8,26)[0,0]),2**-24)
        np.testing.assert_array_equal(aligned_block_product(a[None],b,8,25),
                                      np.zeros((1,1,1),np.float16))
        with self.assertRaises(ValueError):aligned_block_product(a.astype(np.float32),b,8,25)

    def test_demo_duration_mismatch_preserves_motion_and_limits_error_scope(self):
        expected=np.zeros((15,2,7),np.float32);expected[...,6]=1
        actual=np.zeros((13,2,7),np.float32);actual[...,6]=1
        actual[6:,:,0]=100  # Unaligned tail must not enter the reported errors.
        steps=[dict(reference_frames=r,comparisons={'closed_loop':dict(
            frames=a,duration_matches=r==a,pose_token_mismatches=0)})
            for r,a in [(6,6),(6,4),(5,5)]]
        plan=build_plan(0,'closed_loop',expected,actual,steps)
        self.assertFalse(plan['duration_exact'])
        self.assertEqual(plan['aligned_frames'],6)
        self.assertEqual(plan['metrics']['joint_m']['max'],0)
        self.assertEqual(len(plan['per_frame']['root_m']),6)
        self.assertEqual(len(plan['native_joint_positions']),13*2*3)
        steps[0]['comparisons']['closed_loop']['duration_matches']=False
        with self.assertRaises(ValueError):build_plan(0,'closed_loop',expected,actual,steps)
        steps[0]['comparisons']['closed_loop']['frames']=4
        steps[1]['comparisons']['closed_loop']['frames']=6
        steps[1]['comparisons']['closed_loop']['duration_matches']=True
        plan=build_plan(0,'closed_loop',expected,actual,steps)
        self.assertEqual(plan['aligned_frames'],0)
        self.assertIsNone(plan['metrics']['root_m'])
        with self.assertRaises(ValueError):build_plan(0,'closed_loop',expected,actual[:-1],steps)

    def test_mixed_temporal_prefix_requires_explicit_axis_and_valid_mask(self):
        mask_key='graph//mmm_net_1/Less_output_0'
        reference={mask_key:(np.arange(16)<11)[None,:]}
        expected=np.arange(16*5*32).reshape(16,5,32)
        actual=expected[:11].copy()
        result,info=temporal_prefix(reference,'pose','_pose_token_emb',actual,expected,11)
        np.testing.assert_array_equal(result,actual)
        self.assertEqual(info['axis'],0)
        result,info=temporal_prefix(reference,'pose','_pose_token_emb',actual[:10],expected,11)
        self.assertIsNone(info)  # Different predicted duration must not be hidden.
        self.assertEqual(result.shape,expected.shape)
        reference[mask_key][0,15]=True
        with self.assertRaises(ValueError):
            temporal_prefix(reference,'pose','_pose_token_emb',actual,expected,11)
        result,info=temporal_prefix({},'root','unknown',actual,expected,11)
        self.assertIsNone(info)

    def test_attention_prefix_preserves_heads_and_checks_both_temporal_axes(self):
        key='graph//mmm_net/Less_output_0'
        reference={key:(np.arange(16)<8)[None,:]}
        expected=np.arange(16*25*25).reshape(16,25,25)
        name='_root_token_transformer_model.layers.1.self_attn.probabilities'
        actual=expected[:,:17,:17].copy()
        result,info=temporal_prefix(reference,'root',name,actual,expected,8)
        np.testing.assert_array_equal(result,actual)
        self.assertEqual(info['axes'],[1,2])
        for bad in (actual[:15],actual[:,:16],actual[:,:,:16]):
            result,info=temporal_prefix(reference,'root',name,bad,expected,8)
            self.assertIsNone(info)
            self.assertEqual(result.shape,expected.shape)
        reference[key][0,15]=True
        with self.assertRaises(ValueError):
            temporal_prefix(reference,'root',name,actual,expected,8)

    def test_mixed_folded_identity_uses_actual_activation_input(self):
        class Capture(dict):
            @property
            def files(self): return list(self)
        activation='/decoder/model.2.0/model.0/activation1/Relu_output_0'
        conditioned='/decoder/model.2.0/model.0/Mul_output_0'
        reference=Capture({'output/pred_num_tokens':np.array([16]),
            'graph/'+activation:np.ones((1,2,16)),
            'graph/'+conditioned:-np.ones((1,2,16))})
        metadata={'tap.0.name':'soma/12/decoder.model.2.0.model.0.norm1',
                  'tap.0.operation':'VIEW'}
        native={'tap.0':-np.ones((2,16))}
        nodes=[dict(name='activation',op='Relu',inputs=[conditioned],outputs=[activation])]
        report=compare_mixed_layers(reference,metadata,native,nodes)
        self.assertEqual(report['layers'][0]['max_absolute'],0)
        self.assertEqual(report['layers'][0]['reference'],'graph/'+conditioned)
        self.assertEqual(compare_mixed_layers(reference,metadata,native)['compared_taps'],0)
        with self.assertRaises(ValueError):compare_mixed_layers(reference,metadata,native,nodes*2)

    def test_mixed_layers_report_missing_and_mismatched_boundaries(self):
        class Capture(dict):
            @property
            def files(self): return list(self)
        reference=Capture({'output/pred_num_tokens':np.array([16]),
            'graph//mmm_net/_proj_local_pose/Add_output_0':np.zeros((1,8,256)),
            'graph//mmm_net/_proj_local_pose_1/Add_output_0':np.ones((1,64,160))})
        metadata={};native={}
        for i,(index,name,shape) in enumerate([(25,'_proj_local_pose',(8,256)),
                (30,'unknown',(8,256)),(15,'_proj_local_pose',(64,160)),
                (20,'_proj_local_pose',(63,160))]):
            key=f'tap.{i}';metadata[key+'.name']=f'soma/{index}/{name}'
            metadata[key+'.operation']='CPY';native[key]=np.zeros(shape)
        report=compare_mixed_layers(reference,metadata,native)
        self.assertEqual(report['compared_taps'],2)
        self.assertEqual([r['status'] for r in report['layers']],
                         ['compared','unmapped','compared','shape_mismatch'])
        self.assertEqual(report['layers'][2]['component'],'pose')
        self.assertEqual(report['layers'][2]['max_absolute'],1)
        native['tap.2'][0,0]=np.nan
        with self.assertRaises(ValueError):compare_mixed_layers(reference,metadata,native)
        native['tap.2'][0,0]=0
        reference['output/pred_num_tokens']=np.array([5])
        with self.assertRaises(ValueError):compare_mixed_layers(reference,metadata,native)

    def test_duration_selection_respects_masks_and_first_index_ties(self):
        logits=np.zeros(12);logits[1:3]=2;logits[10:]=100
        allowed=np.ones(11);allowed[10]=0
        result=duration_decision(logits,allowed,7)
        self.assertTrue(result['selection_consistent'])
        self.assertEqual(result['runner_up_tokens'],8)
        self.assertEqual(result['winner_margin'],0)
        self.assertFalse(duration_decision(logits,allowed,8)['selection_consistent'])
        allowed[:]=0;allowed[4]=1
        self.assertIsNone(duration_decision(logits,allowed,10)['winner_margin'])
        with self.assertRaises(ValueError):duration_decision(logits,np.zeros(11),10)

    def test_capture_pose_decisions_compare_valid_prefix_only(self):
        reference,actual=motion(),motion()
        key='graph//mmm_net/_proj_pose_output_logit/Add_output_0'
        reference[key]=np.zeros((1,16,80),np.float32)
        actual[key]=reference[key].copy()
        actual[key][0,3,2]=1
        actual[key][0,12,1]=1  # Padded token is not part of the prediction.
        result=compare_pose_decisions(reference,actual)
        self.assertEqual(result['compared_pose_tokens'],40)
        self.assertEqual(result['mismatches'],1)

    def test_local_decision_capture_copies_and_restores_profiler(self):
        def operation(value):
            decision=value+2
            return decision
        capture=TensorCapture();previous=sys.getprofile()
        with ExitStack() as stack:
            capture.capture_locals(stack,operation,'decision',('decision',))
            result=operation(torch.tensor([1.]))
        result.add_(10)
        self.assertIs(sys.getprofile(),previous)
        np.testing.assert_array_equal(capture.tensors['decision/0/decision'],[3.])

    def test_chain_gate_rejects_position_contacts_and_discrete_errors(self):
        transforms=np.zeros((3,2,7),np.float32);transforms[...,6]=1
        expected={'world':transforms,'corrected':transforms}
        actual={key:value.copy() for key,value in expected.items()}
        contacts=np.zeros((3,4),np.float32)
        decisions={'pose_token_count_matches':True,'pose_token_mismatches':0}
        limits={'position_m':1e-4,'rotation_deg':.01,'contacts_absolute':1e-4}
        self.assertTrue(evaluate(expected,actual,contacts,contacts,decisions,limits)['passed'])
        actual['world'][1,1,0]=.001
        result=evaluate(expected,actual,contacts,contacts+.01,
                        dict(decisions,pose_token_mismatches=1),limits)
        self.assertFalse(result['passed'])
        self.assertIn('world: position ceiling exceeded',result['failures'])
        self.assertIn('contact ceiling exceeded',result['failures'])
        self.assertIn('pose-token decisions differ',result['failures'])

    def test_chain_gate_rejects_rotation_and_duration_errors(self):
        transforms=np.zeros((3,2,7),np.float32);transforms[...,6]=1
        expected={'world':transforms,'corrected':transforms}
        actual={key:value.copy() for key,value in expected.items()}
        actual['world'][0,1,3:]=[0,0,np.sin(np.pi/4),np.cos(np.pi/4)]
        actual['corrected']=actual['corrected'][:2]
        contacts=np.zeros((3,4),np.float32)
        result=evaluate(expected,actual,contacts,contacts,
                        {'pose_token_count_matches':True,'pose_token_mismatches':0},
                        {'position_m':1e-4,'rotation_deg':.01,'contacts_absolute':1e-4})
        self.assertFalse(result['passed'])
        self.assertIn('world: rotation ceiling exceeded',result['failures'])
        self.assertIn('corrected: duration or skeleton shape mismatch',result['failures'])

    def test_layer_prefix_requires_known_axis_duration_and_captured_mask(self):
        key='layer/model._vqvae_pose_model.decoder/0/kwargs/token_mask'
        reference={'output/pred_num_tokens':np.array([8]),
                   key:(np.arange(16)<8)[None,:]}
        expected=np.arange(512*32).reshape(512,32)
        actual=expected[:,:16].copy()
        selected,rule=valid_sequence_prefix(reference,'decoder','decoder.model.3.0.model.1.conv1',actual,expected)
        np.testing.assert_array_equal(selected,actual)
        self.assertEqual(rule['axis'],1)
        self.assertEqual(rule['padded_length'],32)
        # Arbitrary module names and wrong sequence lengths remain mismatches.
        for name,value in [('other.module',actual),('decoder.model.3.0.model.1.conv1',actual[:,:15])]:
            selected,rule=valid_sequence_prefix(reference,'decoder',name,value,expected)
            self.assertIsNone(rule)
            self.assertEqual(selected.shape,expected.shape)
        reference[key][0,15]=True
        with self.assertRaisesRegex(ValueError,'temporal mask'):
            valid_sequence_prefix(reference,'decoder','decoder.model.3.0.model.1.conv1',actual,expected)

    def test_root_transformer_prefix_keeps_all_nine_condition_positions(self):
        key='layer/model._mmm_root_model.mmm_net._root_token_transformer_model/0/kwargs/src_key_padding_mask'
        reference={'output/pred_num_tokens':np.array([6]),key:(np.arange(25)>=15)[None,:]}
        expected=np.arange(25*512).reshape(25,512)
        selected,rule=valid_sequence_prefix(reference,'root','_root_token_transformer_model.layers.0.norm1',expected[:15],expected)
        np.testing.assert_array_equal(selected,expected[:15])
        self.assertEqual(rule['valid_length'],15)
        reference[key][0,0]=True
        with self.assertRaisesRegex(ValueError,'temporal mask'):
            valid_sequence_prefix(reference,'root','_root_token_transformer_model.layers.0.norm1',expected[:15],expected)

    def test_capture_copies_before_inplace_mutation_and_restores_method(self):
        class Operation:
            def run(self, value):
                return value.add_(2)
        operation = Operation()
        capture = TensorCapture()
        with ExitStack() as stack:
            capture.wrap(stack, operation, 'run', 'boundary')
            operation.run(torch.tensor([1.0]))
            operation.run(torch.tensor([4.0]))
        self.assertNotIn('run', vars(operation))
        np.testing.assert_array_equal(capture.tensors['boundary/0/input/0'], [1])
        np.testing.assert_array_equal(capture.tensors['boundary/0/output'], [3])
        np.testing.assert_array_equal(capture.tensors['boundary/1/output'], [6])

    def test_world_position_and_rotation_errors(self):
        reference, actual = motion(), motion()
        actual['output/pred_global_motions'][:, :, 0] = .25
        # Rotate every joint 90 degrees around Z in the 6D representation.
        actual['output/pred_global_motions'][:, :, 92:272] = np.tile([0, 1, 0, -1, 0, 0], 30)
        report = compare(reference, actual)['requests'][0]
        self.assertAlmostEqual(report['root_position_m']['max'], .25)
        self.assertAlmostEqual(report['joint_position_m']['max'], .25)
        self.assertAlmostEqual(report['joint_rotation_deg']['max'], 90)

    def test_padding_is_excluded_but_length_mismatch_is_reported(self):
        reference, actual = motion(), motion()
        actual['output/pred_global_motions'][:, 32:] = 100
        actual['output/pred_num_tokens'][0] = 9
        report = compare(reference, actual)
        self.assertEqual(report['token_count_mismatches'], 1)
        self.assertEqual(report['requests'][0]['features_absolute']['max'], 0)

    def test_nonfinite_valid_output_rejected(self):
        reference, actual = motion(), motion()
        actual['output/pred_global_motions'][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            compare(reference, actual)


if __name__ == '__main__':
    unittest.main()
