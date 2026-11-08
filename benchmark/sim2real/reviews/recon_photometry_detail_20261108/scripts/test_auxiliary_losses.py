"""CPU numerical tests against an independent transcription of official MPRNet.

Run from this directory: python3 -B -m unittest -v test_auxiliary_losses
Only tests scalar losses/gradients; never starts training or accesses datasets.
"""

import unittest

import torch
from torch.nn import functional as F

from auxiliary_losses import LaplacianEdgeLoss, PhotometricL1, photometric_l1


def official_rgb_edge_loss(prediction, target):
    """Independent original 3-channel formula, with CPU/dtype adaptation only."""
    k = torch.Tensor([[.05, .25, .4, .25, .05]])
    kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
    kernel = kernel.to(device=prediction.device, dtype=prediction.dtype)

    def conv_gauss(image):
        n_channels, _, kw, kh = kernel.shape
        image = F.pad(image, (kw // 2, kh // 2, kw // 2, kh // 2),
                      mode="replicate")
        return F.conv2d(image, kernel, groups=n_channels)

    def laplacian_kernel(current):
        filtered = conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = conv_gauss(new_filter)
        return current - filtered

    diff = laplacian_kernel(prediction) - laplacian_kernel(target)
    return torch.mean(torch.sqrt(diff * diff + 1e-3 * 1e-3))


class AuxiliaryLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_grayscale_matches_official_repeated_rgb_loss_and_gradients(self):
        generator = torch.Generator().manual_seed(1808)
        for dtype in (torch.float32, torch.float64):
            prediction = torch.rand(2, 1, 9, 11, generator=generator,
                                    dtype=dtype, requires_grad=True)
            target = torch.rand(2, 1, 9, 11, generator=generator,
                                dtype=dtype, requires_grad=True)
            oracle_prediction = prediction.detach().clone().requires_grad_()
            oracle_target = target.detach().clone().requires_grad_()
            actual = LaplacianEdgeLoss(weight=1.0)(prediction, target)
            expected = official_rgb_edge_loss(
                oracle_prediction.repeat(1, 3, 1, 1),
                oracle_target.repeat(1, 3, 1, 1))
            actual_grad = torch.autograd.grad(actual, (prediction, target))
            expected_grad = torch.autograd.grad(
                expected, (oracle_prediction, oracle_target))
            atol, rtol = ((1e-7, 1e-6) if dtype == torch.float32
                          else (1e-14, 1e-12))
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            for grad, oracle_grad in zip(actual_grad, expected_grad):
                torch.testing.assert_close(grad, oracle_grad,
                                           atol=atol, rtol=rtol)

    def test_edge_double_gradcheck(self):
        generator = torch.Generator().manual_seed(19)
        prediction = torch.rand(1, 1, 4, 5, generator=generator,
                                dtype=torch.double, requires_grad=True)
        target = torch.rand(1, 1, 4, 5, generator=generator,
                            dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            LaplacianEdgeLoss(weight=1.0).double(), (prediction, target),
            eps=1e-6, atol=1e-5, rtol=1e-3))

    def test_identical_constant_inputs_have_epsilon_loss_and_zero_gradients(self):
        prediction = torch.full((1, 1, 8, 10), .4, dtype=torch.double,
                                requires_grad=True)
        target = prediction.detach().clone().requires_grad_()
        loss = LaplacianEdgeLoss(weight=1.0).double()(prediction, target)
        gradients = torch.autograd.grad(loss, (prediction, target))
        torch.testing.assert_close(loss, loss.new_tensor(1e-3))
        for gradient in gradients:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient),
                                       atol=0, rtol=0)

    def test_constant_offset_preserves_official_boundary_response(self):
        # MPRNet's finite-image padding does not annihilate a constant offset.
        # Test this explicitly instead of silently changing the filter/borders.
        prediction = torch.full((1, 1, 9, 11), .6, dtype=torch.double,
                                requires_grad=True)
        target = torch.full_like(prediction, .2)
        actual = LaplacianEdgeLoss(weight=1.0).double()(prediction, target)
        expected = official_rgb_edge_loss(
            prediction.repeat(1, 3, 1, 1), target.repeat(1, 3, 1, 1))
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
        gradient, = torch.autograd.grad(actual, prediction)
        self.assertGreater(actual.item(), 1e-3)
        self.assertGreater(gradient.abs().max().item(), 0)

    def test_l1_mean_scale_and_gradients(self):
        prediction = torch.tensor([[[[0., 1.], [.25, .75]]]],
                                  dtype=torch.double, requires_grad=True)
        target = torch.tensor([[[[.5, .5], [.25, 0.]]]],
                              dtype=torch.double, requires_grad=True)
        loss = photometric_l1(prediction, target)
        prediction_grad, target_grad = torch.autograd.grad(loss,
                                                           (prediction, target))
        expected_grad = torch.tensor([[[[-.25, .25], [0., .25]]]],
                                     dtype=torch.double)
        torch.testing.assert_close(loss, loss.new_tensor(.4375))
        torch.testing.assert_close(prediction_grad, expected_grad)
        torch.testing.assert_close(target_grad, -expected_grad)
        torch.testing.assert_close(
            photometric_l1(prediction.repeat(2, 5, 1, 1),
                           target.repeat(2, 5, 1, 1)), loss)
        torch.testing.assert_close(photometric_l1(prediction * 3, target * 3),
                                   loss * 3)

    def test_arbitrary_channels_and_buffer_dtype(self):
        module = LaplacianEdgeLoss(weight=1.0).double()
        self.assertEqual(list(module.parameters()), [])
        self.assertIn("kernel", dict(module.named_buffers()))
        for channels in (1, 3, 5):
            prediction = torch.linspace(0, 1, channels * 7 * 9,
                                        dtype=torch.double).view(1, channels, 7, 9)
            target = prediction.flip(-1)
            self.assertEqual(module(prediction, target).dtype, torch.double)
            torch.testing.assert_close(module(prediction, target),
                                       module(target, prediction))

    def test_reject_mismatched_or_integer_inputs(self):
        prediction = torch.ones(1, 1, 4, 4)
        for loss in (photometric_l1, LaplacianEdgeLoss()):
            with self.assertRaises(ValueError):
                loss(prediction, torch.ones(1, 1, 4, 5))
            with self.assertRaises(TypeError):
                loss(prediction.long(), prediction.long())
            with self.assertRaises(ValueError):
                loss(prediction, prediction.double())

    def test_native_trainer_weights(self):
        prediction = torch.linspace(0, 1, 5 * 7).view(1, 1, 5, 7)
        target = prediction.flip(-1)
        photometric = PhotometricL1()
        edge = LaplacianEdgeLoss()
        self.assertEqual(photometric.weight, 1.0)
        self.assertEqual(edge.weight, .05)
        torch.testing.assert_close(photometric(prediction, target),
                                   photometric_l1(prediction, target))
        torch.testing.assert_close(edge(prediction, target),
                                   .05 * LaplacianEdgeLoss(weight=1)(prediction,
                                                                 target))


if __name__ == "__main__":
    unittest.main(verbosity=2)
