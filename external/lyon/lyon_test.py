import numpy as np
from lyon.calc import LyonCalc


calc = LyonCalc()

print('Testing SOSCASCADE')
inputs = np.array([1, 0, 0, 0, 0, 0], dtype=np.double)
c1 = np.array([1, 0, 0, -.9, 0], dtype=np.double)
c2 = np.array([1, 0, 0, -.8, 0], dtype=np.double)
coeffs = np.hstack([c1[:, np.newaxis], c2[:, np.newaxis]])
out, state = calc.soscascade(inputs, coeffs)
print('Out:')
print(out)

print('Testing AGC')
inputs = np.ones((20, 1), dtype=np.double)
agc_params = np.array([[.5, .5]], dtype=np.double)
out, state = calc.agc(inputs, agc_params)
print('Out:')
print(out)

print('Testing SOSFILTERS')
print('===Test 1')
inputs = np.array([1, 0, 0, 0, 0, 0], dtype=np.double)[:, np.newaxis]
c1 = np.array([1, 0, 0, -.9, 0], dtype=np.double)
c2 = np.array([1, 0, 0, -.8, 0], dtype=np.double)
coeffs = np.hstack([c1[:, np.newaxis], c2[:, np.newaxis]])
out, state = calc.sosfilters(inputs, coeffs)
print('Out:')
print(out)
print('===Test 2')
inputs = np.array([[1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]], dtype=np.double)
inputs = np.ascontiguousarray(inputs.transpose())
out, state = calc.sosfilters(inputs, coeffs)
print('Out:')
print(out)
print('===Test 3')
coeffs = np.array([1, 0, 0, -.9, 0], dtype=np.double)[:, np.newaxis]
out, state = calc.sosfilters(inputs, coeffs)
print('Out:')
print(out)
