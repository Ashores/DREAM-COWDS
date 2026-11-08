import unittest
import torch
from continual_lora.numerics import gamma_zero_focal_loss


class StablePresenceLossTest(unittest.TestCase):
    def test_saturated_finite_analytical_gradient(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for alpha in (0.0, 0.25, 0.5, 1.0, -1.0):
                x = torch.tensor([[-1000., -100., -20., 0., 20., 100., 1000.]],
                                 dtype=dtype, requires_grad=True)
                y = torch.tensor([[0.,1.,0.,1.,0.,1.,1.]])
                value = gamma_zero_focal_loss(x, y, 3., alpha)
                value.backward()
                expected = (x.detach().float().sigmoid() - y) / (3 * x.shape[1])
                if alpha >= 0:
                    expected *= alpha * y + (1-alpha) * (1-y)
                self.assertTrue(torch.isfinite(value).all())
                self.assertTrue(torch.isfinite(x.grad).all())
                torch.testing.assert_close(x.grad, expected.to(dtype), rtol=1e-3, atol=1e-5)

    def test_native_reference_all_reductions(self):
        torch.manual_seed(42)
        for shape in ((3, 7), (2, 3, 4, 5)):
            x = torch.randn(shape, requires_grad=True)
            y = torch.randint(0, 2, shape).float()
            weight = .25*y+.75*(1-y)
            reference = torch.nn.functional.binary_cross_entropy_with_logits(x, y, reduction='none') * weight
            actual = gamma_zero_focal_loss(x, y, 2., reduce=False)
            torch.testing.assert_close(actual, reference)
            for multi in ([False, True] if len(shape) == 4 else [False]):
                expected = reference.flatten(2).mean(-1)/2 if multi else reference.mean(1).sum()/2
                torch.testing.assert_close(gamma_zero_focal_loss(x,y,2.,loss_on_multimask=multi), expected)


if __name__ == '__main__':
    unittest.main()
