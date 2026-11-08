"""CPU rejection fixtures for runtime-only resource adaptation."""
import ast,copy,hashlib,json,unittest
from pathlib import Path
import gpu3_resource_guard as G


def fixture():
    c=json.loads((Path(__file__).parent/'MASK_AUX_RUNTIME_RESOURCE_V2.json').read_text());G.validate_contract(c,G.SCIENCE_SHA)
    s=dict(cuda_visible_devices=G.GPU_UUID,gpu_index=3,gpu_uuid=G.GPU_UUID,gpu_name=G.GPU_NAME,cuda_device_count=1,torch_gpu_name=G.GPU_NAME,
        parent_pid=G.PARENT_PID,parent_start_ticks=int(c['parent_start_ticks']),parent_state='S',parent_cmdline=c['parent_cmdline'].copy(),parent_script_sha256=c['parent_script_sha256'],active_gpu_pids=[G.PARENT_PID,500001])
    return c,s


class GPU3RuntimeContracts(unittest.TestCase):
    def test_exact_frozen_parent_uuid_and_single_device_admitted(self):
        c,s=fixture();r=G.verify_snapshot(c,s,500001);self.assertTrue(r['passed']);self.assertEqual(r['gpu_uuid'],G.GPU_UUID)

    def test_unregistered_gpu_user_rejected_even_if_parent_is_allowed(self):
        c,s=fixture();s['active_gpu_pids'].append(987654)
        with self.assertRaises(ValueError):G.verify_snapshot(c,s,500001)

    def test_wrong_uuid_visibility_type_or_device_count_rejected(self):
        for key,value in [('gpu_uuid','GPU-wrong'),('cuda_visible_devices','3'),('gpu_name','NVIDIA RTX 4090'),('cuda_device_count',2),('torch_gpu_name','NVIDIA TITAN RTX')]:
            c,s=fixture();s[key]=value
            with self.assertRaises(ValueError):G.verify_snapshot(c,s,500001)

    def test_recycled_parent_changed_command_or_source_rejected(self):
        for key,value in [('parent_start_ticks',53273269),('parent_pid',G.PARENT_PID+1),('parent_state','Z'),('parent_cmdline',['python','other.py']),('parent_script_sha256','0'*64)]:
            c,s=fixture();s[key]=value
            with self.assertRaises(ValueError):G.verify_snapshot(c,s,500001)

    def test_actual_string_ticks_and_linux_comm_parentheses_parse_exactly(self):
        c,s=fixture();self.assertEqual(int(c['parent_start_ticks']),53273268)
        fields=['S']+['0']*18+[str(c['parent_start_ticks'])]+['0']*4
        stat=G.parse_proc_stat('335161 (python (queue)) '+' '.join(fields))
        self.assertEqual(stat['start_ticks'],53273268);self.assertEqual(stat['state'],'S')

    def test_guard_is_a_real_separate_admission_not_monkeypatch(self):
        tree=ast.parse((Path(__file__).parent/'run_aux_mask_center.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        context=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='load_context'];self.assertEqual(len(context),1)
        self.assertTrue(any(k.arg=='check_gpu' and isinstance(k.value,ast.Constant) and k.value.value is False for k in context[0].keywords))
        guards=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='check_resource'];self.assertEqual(len(guards),3)
        self.assertNotIn('E.check_gpu=',(Path(__file__).parent/'run_aux_mask_center.py').read_text())


if __name__=='__main__':unittest.main(verbosity=2)
