from scipy.interpolate import interp1d
import numpy as np
import glob
import os


class dMdEdist():
    def __init__(self):
        self.betas = []
        self.masses = []
        self.functions = []

        script_path = os.path.abspath(__file__)  # i.e./path/to/dir/dmde_values
        script_dir = os.path.split(script_path)[0]  # i.e./path/to/dir/
        rel_path = "dmde_values/*.dat"
        abs_file_path = os.path.join(script_dir, rel_path)

        for file in glob.glob(abs_file_path):
            self.betas.append(float(file.split('/')[-1].split('_')[0]))
            with open(file, 'r') as f:
                a = []
                b = []
                x = []
                y = []
                for i, line in enumerate(f):
                    if i == 1:
                        self.masses.append(float(line.split()[0]) / 2.0)
                    elif i == 3:
                        a = [float(string) for string in line.split()]
                    elif i == 4:
                        b = [float(string) for string in line.split()]
                    else:
                        continue

                for i, point in enumerate(a):
                    if point > 0:
                        x.append(a[i])
                        y.append(b[i])

                n = len(x)
                step = 10
                s = []
                p_s = []

                # For smoothing!
                nbins = int(n/step)
                for i in range(nbins):
                    xsum = sum(x[step * i:(step * i) + step - 1])
                    s_point = xsum / float(step)
                    s.append(s_point)

                    ysum = sum(y[step * i:(step * i) + step - 1])
                    p_s_point = ysum / float(step)
                    p_s.append(p_s_point)
                s.append(0.0)
                p_s.append(0.0)
                s = s[::-1]
                p_s = p_s[::-1]

                CDE = np.cumsum(p_s) / max(np.cumsum(p_s))
                f = interp1d(CDE, s)
                self.functions.append(f)

    def beta_mass_interp(self):
        f = interp1d(self.betas, self.masses)
        return f

    def beta_interp(self, beta):
        diffs = np.array([np.absolute(b - beta) for b in self.betas])
        [i, j] = diffs.argsort()[:2]

        x1 = self.betas[i]
        x2 = self.betas[j]

        f1 = self.functions[i]
        f2 = self.functions[j]

        d1 = diffs[i]
        d2 = diffs[j]
        d = d1 + d2

        p2 = d1 / d
        p1 = d2 / d

        def f_new(x):
            return p1 * f1(x) + p2 * f2(x)

        l = [f_new, x1, x2]
        return l

    def energy_spread(self, beta, nfrag):
        [beta_func, x1, x2] = self.beta_interp(beta)
        bin_size = 1.0 / nfrag
        half = bin_size * 0.5
        x_set = np.linspace(half, 1.0 - half, nfrag)
        return beta_func(x_set)

# print('dMdE dist file imported')
