import matplotlib.pyplot as plt
import numpy as np
from sigmoid_module import logit, sigmoid
import os.path as p
def target_file_abs_path(target_file_relative_path):
    return p.join(p.dirname(p.realpath(__file__)), target_file_relative_path)

print(logit(0.5))
print(sigmoid(0.5))


z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)


plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel("$\phi (z)$")
plt.yticks([0.0, 0.5, 1.0])

ax = plt.gca()
ax.yaxis.grid(True)

plt.savefig(target_file_abs_path('./sigmoid_curve.png'))