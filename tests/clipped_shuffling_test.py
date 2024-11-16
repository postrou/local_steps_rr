import unittest

import numpy as np

from src.optimizers import ClERR, ClERR2, ClippedShuffling
from src.loss_functions import load_fourth_order_dataset


class ClERRTestCase(unittest.TestCase):
    pass
    # def test_gradient_norm_clerr_2_full_batch_crr_full_batch(self):
    #     n_epochs = 100
    #     n_seeds = 2
    #     step_size = 0.001
    #     clip_level = 10000

    #     loss, x0, x_opt, n_seeds, stoch_it, trace_len, _, _ = \
    #         load_fourth_order_dataset(n_epochs, n_seeds, save_results=False)

    #     batch_size = loss.n
    #     trace_len = n_epochs
    #     c_0 = 1 / (2 * step_size)
    #     c_1 = c_0 / clip_level
    #     inner_step_size = 0
    #     use_g = False
    #     clerr_2 = ClERR2(
    #         c_0=c_0,
    #         c_1=c_1,
    #         inner_step_size=inner_step_size,
    #         loss=loss, 
    #         it_max=stoch_it,
    #         batch_size=batch_size, 
    #         trace_len=trace_len,
    #         n_seeds=n_seeds, 
    #         use_g_in_outer_step=use_g
    #     )
    #     clerr_2_trace = clerr_2.run(x0=x0)
    #     clerr_2_trace.convert_its_to_epochs(batch_size=batch_size)
    #     clerr_2_trace.compute_loss_of_iterates()

    #     batch_size_crr = loss.n
    #     crr = ClippedShuffling(
    #         steps_per_permutation=np.inf,
    #         lr0=step_size,
    #         clip_level=clip_level,
    #         loss=loss, 
    #         it_max=n_epochs, 
    #         batch_size=batch_size_crr, 
    #         trace_len=trace_len,
    #         n_seeds=n_seeds, 
    #     )
    #     crr_trace = crr.run(x0=x0)
    #     crr_trace.convert_its_to_epochs(batch_size=loss.n)
    #     crr_trace.compute_loss_of_iterates()

    #     for i_seed, seed in enumerate(crr_trace.grad_estimators_norms_all):
    #         self.assertEqual(len(crr_trace.grad_estimators_norms_all[seed]), trace_len + 1, \
    #             f'seed #{i_seed}')
    #         for i in range(n_epochs):
    #             self.assertEqual(i, clerr_2_trace.its_all[seed][i], \
    #                 f'seed #{i_seed}, epoch #{i}')
    #             crr_full_grad = crr_trace.grad_estimators_norms_all[seed][i]
    #             clerr_2_full_grad = clerr_2_trace.grad_estimators_norms_all[seed][i]
    #             self.assertAlmostEqual(
    #                 loss.norm(crr_full_grad), 
    #                 loss.norm(clerr_2_full_grad), 
    #                 places=4,
    #                 msg=f'seed #{i_seed}, epoch #{i}'
    #             )

    # def test_gradient_norm_clerr_2_crr_full_batch_on_epochs(self):
    #     n_epochs = 100
    #     n_seeds = 2
    #     batch_size = 32
    #     step_size = 0.001
    #     clip_level = 10000

    #     loss, x0, x_opt, n_seeds, stoch_it, trace_len, _, _ = \
    #         load_fourth_order_dataset(n_epochs, n_seeds, batch_size, save_results=False)

    #     trace_len = n_epochs
    #     c_0 = 1 / (2 * step_size)
    #     c_1 = c_0 / clip_level
    #     inner_step_size = 0
    #     use_g = False
    #     clerr_2 = ClERR2(
    #         c_0=c_0,
    #         c_1=c_1,
    #         inner_step_size=inner_step_size,
    #         loss=loss, 
    #         it_max=stoch_it,
    #         batch_size=batch_size, 
    #         trace_len=trace_len,
    #         n_seeds=n_seeds, 
    #         use_g_in_outer_step=use_g
    #     )
    #     clerr_2_trace = clerr_2.run(x0=x0)
    #     clerr_2_trace.convert_its_to_epochs(batch_size=batch_size)
    #     clerr_2_trace.compute_loss_of_iterates()

    #     batch_size_crr = loss.n
    #     crr = ClippedShuffling(
    #         steps_per_permutation=np.inf,
    #         lr0=step_size,
    #         clip_level=clip_level,
    #         loss=loss, 
    #         it_max=n_epochs, 
    #         batch_size=batch_size_crr, 
    #         trace_len=trace_len,
    #         n_seeds=n_seeds, 
    #     )
    #     crr_trace = crr.run(x0=x0)
    #     crr_trace.convert_its_to_epochs(batch_size=loss.n)
    #     crr_trace.compute_loss_of_iterates()

    #     for i_seed, seed in enumerate(crr_trace.grad_estimators_norms_all):
    #         self.assertEqual(len(crr_trace.grad_estimators_norms_all[seed]), trace_len + 1, \
    #             f'seed #{i_seed}')
    #         for i in range(n_epochs):
    #             self.assertEqual(i, clerr_2_trace.its_all[seed][i], \
    #                 f'seed #{i_seed}, epoch #{i}')
    #             crr_full_grad = crr_trace.grad_estimators_norms_all[seed][i]
    #             clerr_2_full_grad = clerr_2_trace.grad_estimators_norms_all[seed][i]
    #             self.assertAlmostEqual(
    #                 loss.norm(crr_full_grad), 
    #                 loss.norm(clerr_2_full_grad), 
    #                 places=4,
    #                 msg=f'seed #{i_seed}, epoch #{i}'
    #             )

                    
if __name__ == '__main__':
    unittest.main()