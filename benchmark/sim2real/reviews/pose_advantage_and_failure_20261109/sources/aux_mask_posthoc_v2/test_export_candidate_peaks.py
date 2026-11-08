"""CPU-only native readout/region classification contracts, no trained-model run."""
import hashlib,tempfile,unittest
from pathlib import Path
import numpy as np
from export_candidate_peaks import category,decode_original_probability,verify_control_reference,checked_lineage


class CandidatePeakExportTests(unittest.TestCase):
    def test_original_probability_readout_is_not_changed(self):
        p=np.zeros((16,16),np.float32);p[2,5]=.75;p[10,13]=.25
        uv=np.array([(.75*5.5+.25*13.5)/16,(.75*2.5+.25*10.5)/16],np.float32);before=p.copy();r=decode_original_probability(p,uv)
        self.assertEqual((r['row'],r['column']),(2,5));self.assertEqual(r['uv'],[5.5/16,2.5/16]);self.assertTrue(np.array_equal(p,before))

    def test_uniform_ties_keep_original_first_native_grid_point(self):
        p=np.full((16,16),1/256,np.float32);r=decode_original_probability(p,[.5,.5]);self.assertEqual(r['flat_index'],0)

    def test_point_miss_is_not_called_wrong_object_if_cell_has_support(self):
        no=dict(available=True,peak_in_region=False);yes=dict(available=True,peak_in_region=True)
        self.assertEqual(category(yes,yes,.1,.2),'native_peak_in_visible_target')
        self.assertEqual(category(no,no,.01,.1),'native_point_outside_endpoint_but_peak_cell_has_target_support')
        self.assertEqual(category(no,no,0.,.01),'no_endpoint_cell_support_but_endpoint_union_cell_support')
        self.assertEqual(category(no,no,0.,0.),'no_target_support_in_endpoint_or_union_peak_cell')
        self.assertEqual(category(dict(available=False,peak_in_region=None),no,0.,.2),'region_unavailable_or_empty')

    def test_probability_mismatch_or_nan_is_fatal(self):
        p=np.full((16,16),1/256,np.float32)
        with self.assertRaises(ValueError):decode_original_probability(p,[.4,.5])
        p[0,0]=np.nan
        with self.assertRaises(ValueError):decode_original_probability(p,[.5,.5])

    def test_equal_uv_with_different_native_peak_is_rejected(self):
        a=np.zeros((16,16),np.float32);a[1,1]=a[14,14]=.5
        b=np.zeros((16,16),np.float32);b[1,14]=b[14,1]=.5
        pa=decode_original_probability(a,[.5,.5]);pb=decode_original_probability(b,[.5,.5])
        reference=dict(pred_uv=[.5,.5],center_argmax_uv=pa['uv'],alt_uv_peak=pa['uv'])
        self.assertTrue(verify_control_reference([.5,.5],pa,reference)['native_peak_exact'])
        # Both satisfy the continuous UV check; the different native peak must fail.
        with self.assertRaises(AssertionError):verify_control_reference([.5,.5],pb,reference)

    def test_same_size_raw_source_mutation_is_rejected_by_actual_byte_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'fixture_input.npy';p.write_bytes(b'original raw bytes')
            r=dict(path=str(p),size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(checked_lineage([r]),[r]);p.write_bytes(b'changed! raw bytes')
            self.assertEqual(p.stat().st_size,r['size_bytes'])
            with self.assertRaises(ValueError):checked_lineage([r])


if __name__=='__main__':unittest.main(verbosity=2)
