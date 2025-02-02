import cirq
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
import scipy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random


def R(fi, hi, i=0, j=1):
    N = 3
    if i == j:
        return np.eye(N)
    if i > j:
        i, j = j, i
    x_for_ms = np.zeros((N, N))
    x_for_ms[i][j] = 1
    x_for_ms[j][i] = 1
    y_for_ms = np.zeros((N, N))
    y_for_ms[i][j] = -1
    y_for_ms[j][i] = 1
    y_for_ms = y_for_ms * 1j

    m = np.cos(fi) * x_for_ms + np.sin(fi) * y_for_ms

    return linalg.expm(-1j * m * hi / 2)
def make_ms_matrix(N, fi, hi, i, j, k, l):
    if i == j:
        return np.eye(N)
    if i > j:
        i, j = j, i
    x_for_ms1 = np.zeros((N, N))
    x_for_ms1[i][j] = 1
    x_for_ms1[j][i] = 1
    y_for_ms1 = np.zeros((N, N))
    y_for_ms1[i][j] = -1
    y_for_ms1[j][i] = 1
    y_for_ms1 = 1j * y_for_ms1
    if k == l:
        return
    if k > l:
        k, l = l, k
    x_for_ms2 = np.zeros((N, N))
    x_for_ms2[k][l] = 1
    x_for_ms2[l][k] = 1
    y_for_ms2 = np.zeros((N, N))
    y_for_ms2[k][l] = -1
    y_for_ms2[l][k] = 1
    y_for_ms1 = 1j * y_for_ms1

    m = np.kron((np.cos(fi) * x_for_ms1 + np.sin(fi) * y_for_ms1), (np.cos(fi) * x_for_ms2 + np.sin(fi) * y_for_ms2))
    m = -1j * m * hi
    return linalg.expm(m)

class TwoQS(gate_features.TwoQubitGate
                      ):

    def __init__(self, coaf, diag_i='XX'):
        #self.mat = mat
        self.diag_info = diag_i
        self.coaf = coaf

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__
        }

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        return cls()

    def _qid_shape_(self):
        return (3, 3,)

    def _unitary_(self):
        matrix = make_ms_matrix(3, 0, -np.pi / 2,self.coaf[0],self.coaf[1],self.coaf[2],self.coaf[3])
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

class U(cirq.Gate):
    def __init__(self, mat, diag_i='R'):
        self.mat = mat
        self.diag_info = diag_i

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return self.mat

    def _circuit_diagram_info_(self, args):
        return self.diag_info

class X1(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'X1'

x01 = X1()
def encoding(cirquit, q_mask, v, vb):
    for o in range(B):

        u11 = U(R(0, np.pi/2 + (v[0 + o * 4] // (np.pi/8)) * (np.pi/8), 1, 2), 'Rx(-π)12')
        u21 = U(R(0, np.pi/2 + (v[1 + o * 4] // (np.pi/8)) * (np.pi/8), 1, 2), 'Rx(-π)12')
        u12 = U(R(0, np.pi/2 + (v[2 + o * 4] // (np.pi/8)) * (np.pi/8), 0, 2), 'Rx(-π)12')
        u22 = U(R(0, np.pi/2 + (v[3 + o * 4] // (np.pi/8)) * (np.pi/8), 0, 2), 'Rx(-π)12')

        u1 = U(R(vb[0], vb[1],0, 1), 'Rx(-π)12')
        u2 = U(R(vb[2], vb[3], 0, 1), 'Ry(π/2)01')
        u3 = U(R(vb[4], vb[5], 0, 1), 'Ry(-π/2)01')
        u4 = U(R(vb[6], vb[7], 0, 1), 'Rx(-π)01')

        u11r = U(R(0, - np.pi/2 + (v[0 + o * 4] // (np.pi/8)) * (np.pi/8), 1, 2), 'Rx(-π)12')
        u21r = U(R(0, - np.pi/2 + (v[1 + o * 4] // (np.pi/8)) * (np.pi/8), 1, 2), 'Rx(-π)12')
        u12r = U(R(0, - np.pi/2 + (v[2 + o * 4] // (np.pi/8)) * (np.pi/8), 0, 2), 'Rx(-π)12')
        u22r = U(R(0, - np.pi/2 + (v[3 + o * 4] // (np.pi/8)) * (np.pi/8), 0, 2), 'Rx(-π)12')

        xx = TwoQS([0, 1, 0, 1])
        cirquit.append([u11(q_mask[o][0]), u21(q_mask[o][1])], strategy=InsertStrategy.INLINE)
        cirquit.append([u12(q_mask[o][0]), u22(q_mask[o][1])], strategy=InsertStrategy.INLINE)

        cirquit.append([u1(q_mask[o][0])], strategy=InsertStrategy.INLINE)
        cirquit.append([xx(q_mask[o][0], q_mask[o][1])], strategy=InsertStrategy.INLINE)
        cirquit.append([u2(q_mask[o][0]), u4(q_mask[o][1])], strategy=InsertStrategy.INLINE)
        cirquit.append([u3(q_mask[o][0])], strategy=InsertStrategy.INLINE)

        cirquit.append([u12r(q_mask[o][0]), u22r(q_mask[o][1])], strategy=InsertStrategy.INLINE)
        cirquit.append([u11r(q_mask[o][0]), u21r(q_mask[o][1])], strategy=InsertStrategy.INLINE)

def operation(v):


    # Проходимся по всем 8 базисным состояниям из кубитного подпр-ва
    ans = 0
    bobr = 0
    ''' 
    for o in range(B):
        bobr += abs((v[0 + o * 4] // (np.pi/4)) * (np.pi/4) - v[0 + o * 4])
        bobr += abs((v[1 + o * 4] // (np.pi/4)) * (np.pi/4) - v[1 + o * 4])
        bobr += abs((v[2 + o * 4] // (np.pi/4)) * (np.pi/4) - v[2 + o * 4])
        bobr += abs((v[3 + o * 4] // (np.pi/4)) * (np.pi/4) - v[3 + o * 4])
    '''

    for i1 in range(2):
        for i2 in range(2):
            for i3 in range(2):

                circuit1 = cirq.Circuit()
                qutrits1 = []
                qutrits1.append(cirq.LineQid(0, dimension=3))
                qutrits1.append(cirq.LineQid(1, dimension=3))
                qutrits1.append(cirq.LineQid(2, dimension=3))

                q1, q2, q3 = qutrits1[0], qutrits1[1], qutrits1[2]

                # Q_mask = [(q1, q3), (q3, q1), (q2, q3)] # для 3 MS-ов
                # Q_mask = [(q1, q3), (q3, q1), (q2, q3), (q3, q2), (q1, q2)] # для 4 MS-ов
                Q_mask = [(q1, q3), (q3, q1), (q2, q3), (q3, q2), (q1, q2), (q2, q1)] # для 6 MS-ов

                for j in range(i1):
                    circuit1.append([x01(q1)], strategy=InsertStrategy.INLINE)
                for j in range(i2):
                    circuit1.append([x01(q2)], strategy=InsertStrategy.INLINE)
                for j in range(i3):
                    circuit1.append([x01(q3)], strategy=InsertStrategy.INLINE)

                vb = [np.pi/2, np.pi/2, 0, -np.pi, np.pi/2, -np.pi/2, 0, -np.pi]

                encoding(circuit1, Q_mask, v, vb)

                ro_ab = cirq.final_state_vector(circuit1, qubit_order=qutrits1)
                ans += abs((ro_ab[9:] * np.conj(ro_ab[9:])).sum())

    plotus.append(ans/8 + bobr/(24*np.pi))
    print(ans / 8)

    return ans / 8 + bobr/(24 * 4 * np.pi)


plotus = []
B = 6
dln = 8

# guess вар1
guess = np.zeros((1, 24))[0]

# Добавляем рандомные сдвиги к элементам guess
delta = np.pi

''' 
for i in range(24):
    guess[i] = guess[i] + np.pi/2
'''

guess = np.array(guess)

bnds = []

for i in range(24):
    bnds.append((-100, 100))

options = {'maxiter': 1000}

res1 = scipy.optimize.minimize(operation, guess, method='COBYLA', bounds=bnds, options=options)

print(res1)
print(list(res1.x))
print()
#print(guess)
plx = np.arange(0, len(plotus))
plt.scatter(plx, plotus, s = 5)
plt.show()