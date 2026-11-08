"""Standard-library exact-source derivation checks; no model/data imports."""
import ast,hashlib,json,unittest
from pathlib import Path


def verify_derivation(original,derived):
    replacements=[(b'x, y = self.shared.affine(normal[idx], target[idx], rng)',b'x, raw_teacher_x, y = self.native.matched_affine(normal[idx], self.raw_normal[idx], target[idx], rng)'),
                  (b'reference = teacher(x)',b'reference = teacher(raw_teacher_x)')]
    if hashlib.sha256(original).hexdigest()!='e1883b68715ecf067a9f4dccbd7a61af5f5dc26b557e061859f4066e7de8d38c':raise ValueError('Original source changed')
    expected=original
    for before,after in replacements:
        if expected.count(before)!=1:raise ValueError('Nonunique replacement')
        expected=expected.replace(before,after)
    if derived!=expected:raise ValueError('Derived source differs beyond the two registered replacements')
    recovered=derived
    for before,after in reversed(replacements):recovered=recovered.replace(after,before)
    if recovered!=original:raise ValueError('Reverse replacement failed')
    ast.parse(derived.decode());return True


class DerivationTests(unittest.TestCase):
    def test_exact_two_registered_replacements_and_ast(self):
        p=Path(__file__).parent;self.assertTrue(verify_derivation((p/'pose_priority_policy.py').read_bytes(),(p/'raw_teacher_policy.py').read_bytes()))
        metadata=json.loads((p/'POLICY_DERIVATION.json').read_text())
        self.assertEqual(metadata['derived_sha256'],hashlib.sha256((p/'raw_teacher_policy.py').read_bytes()).hexdigest())
        self.assertTrue(metadata['other_source_bytes_unchanged'])
    def test_optimizer_or_loss_changes_are_rejected(self):
        p=Path(__file__).parent;original=(p/'pose_priority_policy.py').read_bytes();derived=(p/'raw_teacher_policy.py').read_bytes()
        for before,after in [(b'lr=.00015',b'lr=.0003'),(b'2 * clean_term',b'0 * clean_term')]:
            self.assertIn(before,derived)
            with self.assertRaises(ValueError):verify_derivation(original,derived.replace(before,after))


if __name__=='__main__':unittest.main(verbosity=2)
