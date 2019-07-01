import os
import os.path
import unittest

from cmdstanpy import TMPDIR
from cmdstanpy.lib import Model, OptimizeArgs

datafiles_path = os.path.join("test", "data")


class OptimizeArgsTest(unittest.TestCase):
    def test_args_seed_int(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        args = OptimizeArgs(model,
                            seed=123,
                            output_file=output)
        cmd = args.compose_command('output')
        self.assertIn("seed=123", cmd)

    def test_args_seed_list(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        self.assertRaises(ValueError, lambda: OptimizeArgs(model, seed=[123], output_file=output))

    def test_args_algorithm(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')

        self.assertRaises(ValueError, lambda: OptimizeArgs(model, algorithm="xxx", output_file=output))
        args = OptimizeArgs(model,
                            algorithm="Newton",
                            output_file=output)
        cmd = args.compose_command('output')
        self.assertIn("algorithm=newton", cmd)

    def test_args_algorithm_init_alpha(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        args = OptimizeArgs(model, algorithm="BFGS", init_alpha=2e-4, output_file=output)

        cmd = args.compose_command('output')
        self.assertIn("init_alpha=0.0002", cmd)
        self.assertRaises(ValueError, lambda: OptimizeArgs(model, init_alpha=-1, output_file=output))

    def test_args_algorithm_iter(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        args = OptimizeArgs(model, iter=300, output_file=output)
        cmd = args.compose_command('output')
        self.assertIn("iter=300", cmd)
        self.assertRaises(ValueError, lambda: OptimizeArgs(model, iter=-1, output_file=output))

    def test_args_inits_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        jinits = os.path.join(datafiles_path, 'bernoulli.init.json')
        args = OptimizeArgs(model,
                            data=jdata,
                            inits=jinits)
        cmd = args.compose_command('output')
        s1 = 'data file=test/data/bernoulli.data.json init=test/data/bernoulli.init.json'
        self.assertIn(s1, cmd)
        self.assertRaises(ValueError, lambda: OptimizeArgs(model, data=jdata, inits=[jinits]))


if __name__ == '__main__':
    unittest.main()
