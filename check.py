import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
import scipy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random


X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
Y = np.array([[0, complex(0, -1), 0], [complex(0, 1), 0, 0], [0, 0, 1]])
Z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
id = np.eye(3)

z = np.array([[1, 0, 0]]).T
e = np.array([[0, 1, 0]]).T
f = np.array([[0, 0, 1]]).T
basis = [z, e, f]
paulies1 = [id, X, Y, Z]


def dag(matrix):
    return np.conj(matrix.T)

def nice_repr(parameter):
    """Nice parameter representation
        SymPy symbol - as is
        float number - 3 digits after comma
    """
    if isinstance(parameter, float):
        return f'{parameter:.3f}'
    else:
        return f'{parameter}'

def levels_connectivity_check(l1, l2):
    """Check ion layers connectivity for gates"""
    connected_layers_list = [{0, i} for i in range(max(l1, l2) + 1)]
    assert {l1, l2} in connected_layers_list, "Layers are not connected"

def generalized_sigma(index, i, j, dimension=4):
    """Generalized sigma matrix for qudit gates implementation"""

    sigma = np.zeros((dimension, dimension), dtype='complex')

    if index == 0:
        # identity matrix elements
        sigma[i][i] = 1
        sigma[j][j] = 1
    elif index == 1:
        # sigma_x matrix elements
        sigma[i][j] = 1
        sigma[j][i] = 1
    elif index == 2:
        # sigma_y matrix elements
        sigma[i][j] = -1j
        sigma[j][i] = 1j
    elif index == 3:
        # sigma_z matrix elements
        sigma[i][i] = 1
        sigma[j][j] = -1

    return sigma

class QuditGate(cirq.Gate):
    """Base class for qudits gates"""

    def __init__(self, dimension=4, num_qubits=1):
        self.d = dimension
        self.n = num_qubits
        self.symbol = None

    def _num_qubits_(self):
        return self.n

    def _qid_shape_(self):
        return (self.d,) * self.n

    def _circuit_diagram_info_(self, args):
        return (self.symbol,) * self.n

class QuditRGate(QuditGate):
    """Rotation between two specified qudit levels: l1 and l2"""

    def __init__(self, l1, l2, theta, phi, dimension=4):
        super().__init__(dimension=dimension)
        levels_connectivity_check(l1, l2)
        self.l1 = l1
        self.l2 = l2
        self.theta = theta
        self.phi = phi

    def _unitary_(self):
        sigma_x = generalized_sigma(1, self.l1, self.l2, dimension=self.d)
        sigma_y = generalized_sigma(2, self.l1, self.l2, dimension=self.d)

        s = np.sin(self.phi)
        c = np.cos(self.phi)

        u = scipy.linalg.expm(-1j * self.theta / 2 * (c * sigma_x + s * sigma_y))

        return u

    def _is_parameterized_(self) -> bool:
        return cirq.protocols.is_parameterized(any((self.theta, self.phi)))

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool):
        return self.__class__(self.l1, self.l2, resolver.value_of(self.theta, recursive),
                              resolver.value_of(self.phi, recursive), dimension=self.d)

    def _circuit_diagram_info_(self, args):
        self.symbol = 'R'
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return f'{self.symbol}{str(self.l1).translate(SUB)}{str(self.l2).translate(SUP)}' + f'({nice_repr(self.theta)}, {nice_repr(self.phi)})'

class QuditXXGate(QuditGate):
    """Two qudit rotation for two specified qudit levels: l1 and l2"""

    def __init__(self, l1, l2, theta, dimension=4):
        levels_connectivity_check(l1, l2)
        super().__init__(dimension=dimension, num_qubits=2)
        self.l1 = l1
        self.l2 = l2
        self.theta = theta

    def _unitary_(self):
        sigma_x = generalized_sigma(1, self.l1, self.l2, dimension=self.d)
        u = scipy.linalg.expm(-1j * self.theta / 2 * np.kron(sigma_x, sigma_x))

        return u

    def _is_parameterized_(self) -> bool:
        return cirq.protocols.is_parameterized(self.theta)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool):
        return self.__class__(self.l1, self.l2, resolver.value_of(self.theta, recursive), dimension=self.d)

    def _circuit_diagram_info_(self, args):
        self.symbol = 'XX'
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        info = f'{self.symbol}{str(self.l1).translate(SUB)}{str(self.l2).translate(SUP)}'.translate(
            SUB) + f'({nice_repr(self.theta)})'
        return info, info

class QuditZZGate(QuditGate):
    """Two qudit rotation for two specified qudit levels: l1 and l2"""

    def __init__(self, l1, l2, theta, dimension=4):
        levels_connectivity_check(l1, l2)
        super().__init__(dimension=dimension, num_qubits=2)
        self.l1 = l1
        self.l2 = l2
        self.theta = theta

    def _unitary_(self):
        sigma_z = generalized_sigma(3, self.l1, self.l2, dimension=self.d)
        u = scipy.linalg.expm(-1j * self.theta / 2 * np.kron(sigma_z, sigma_z))

        return u

    def _is_parameterized_(self) -> bool:
        return cirq.protocols.is_parameterized(self.theta)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool):
        return self.__class__(self.l1, self.l2, resolver.value_of(self.theta, recursive), dimension=self.d)

    def _circuit_diagram_info_(self, args):
        self.symbol = 'ZZ'
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        info = f'{self.symbol}{str(self.l1).translate(SUB)}{str(self.l2).translate(SUP)}'.translate(
            SUB) + f'({nice_repr(self.theta)})'
        return info, info

class QuditBarrier(QuditGate):
    """Just barrier for visual separation in circuit diagrams. Does nothing"""

    def __init__(self, dimension=4, num_qudits=2):
        super().__init__(dimension=dimension, num_qubits=num_qudits)
        self.symbol = '|'

    def _unitary_(self):
        return np.eye(self.d * self.d)

class QuditArbitraryUnitary(QuditGate):
    """Random unitary acts on qubits"""

    def __init__(self, dimension=4, num_qudits=2):
        super().__init__(dimension=dimension, num_qubits=num_qudits)
        self.unitary = np.array(scipy.stats.unitary_group.rvs(self.d ** self.n))
        self.symbol = 'U'

    def _unitary_(self):
        return self.unitary

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

class TwoQuditMSGate3_c(gate_features.TwoQubitGate
                        ):

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
        matrix = make_ms_matrix(0, -np.pi / 2)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101_c',
                          'XX0101_c'))

class TwoQuditMSGate02(gate_features.TwoQubitGate
                      ):

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
        matrix = make_ms_matrix(3, 0, np.pi / 2,0,1,0,2)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

class TwoQuditMSGate01(gate_features.TwoQubitGate
                      ):

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
        matrix = make_ms_matrix(3, 0, np.pi / 2,0,1,0,1)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

class unit3(gate_features.ThreeQubitGate
                      ):

    def __init__(self, mat, diag_i='R'):
        self.mat = mat
        self.diag_info = diag_i

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__
        }

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        return cls()

    def _qid_shape_(self):
        return (3, 3, 3)

    def _unitary_(self):
        matrix = self.mat
        return matrix

    def num_controls(self):
        return 3

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

class TwoQuditMSGate12(gate_features.TwoQubitGate
                      ):

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
        matrix = make_ms_matrix(3, 0, np.pi / 2, 0,1,1,2)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

class TwoQuditMSGate01_c(gate_features.TwoQubitGate
                      ):

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
        matrix = make_ms_matrix(3, 0, -np.pi / 2,0,1,0,1)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))
#xx = TwoQS([0,1,0,1])
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


class rTwoQS(gate_features.TwoQubitGate
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
        matrix = make_ms_matrix(3, 0, np.pi / 2,self.coaf[0],self.coaf[1],self.coaf[2],self.coaf[3])
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))



class rTwoQuditMSGate01(gate_features.TwoQubitGate
                      ):

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
        matrix = make_ms_matrix(3, 0, -np.pi / 2,0,1,0,1)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

class rTwoQuditMSGate02(gate_features.TwoQubitGate
                      ):

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
        matrix = make_ms_matrix(3, 0, -np.pi / 2,0,1,0,2)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

class rTwoQuditMSGate12(gate_features.TwoQubitGate
                      ):

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
        matrix = make_ms_matrix(3, 0, -np.pi / 2,0,1,1,2)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))

class U_press(gate_features.TwoQubitGate
              ):

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__
        }

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        return cls()

    def _qid_shape_(self):
        return (3, 3, 3,)

    def _unitary_(self):
        matrix = make_ms_matrix(0, np.pi / 2)
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

def U1_clear(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u6 = U(R(np.pi / 2, -np.pi, 0, 2), 'Ry(-π)02')
    xx = TwoQuditMSGate01()
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)

def rU1_clear(cirquit, q1, q2):
    u1 = U(R(0, np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(π/2)01')
    u6 = U(R(np.pi / 2, np.pi, 0, 2), 'Ry(-π)02')
    xx = rTwoQuditMSGate01()
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)

def U1_c_clear(cirquit, q1, q2):

    u1 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    u2 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u6 = U(R(np.pi / 2, np.pi, 0, 2), 'Ry(π)02')
    xx_c = TwoQuditMSGate01_c()
    cirquit.append([xx_c(q1, q2)], strategy=InsertStrategy.INLINE)
    # adde(cirquit, [xx_c], [q1, q2], 2)
    # error(cirquit, [q1, q2], PMS)
    # adde(cirquit, [u2], [q1], 1)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    # adde(cirquit, [u1, u6], [q1, q2], 1)
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)

def CX_clear01(cirquit, q1, q2):
    u1 = U(R(np.pi/2, np.pi/2, 0, 1), 'Rx(-π)12')
    u2 = U(R(0, - np.pi , 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 0, 1), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    xx = TwoQuditMSGate01()
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)

def CX_clear02(cirquit, q1, q2):
    u1 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Rx(-π)12')
    u2 = U(R(0, - np.pi, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 0, 2), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    xx = TwoQuditMSGate02()
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)

def CX_clear12(cirquit, q1, q2):
    u1 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Rx(-π)12')
    u2 = U(R(0, - np.pi, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 1, 2), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    xx = TwoQuditMSGate12()
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)

def rCX_clear12(cirquit, q1, q2):
    u1 = U(R(np.pi / 2, - np.pi / 2, 0, 1), 'Rx(-π)12')
    u2 = U(R(0, np.pi, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, np.pi, 1, 2), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(-π/2)01')
    xx = rTwoQuditMSGate12()
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)

def rCX_clear01(cirquit, q1, q2):
    u1 = U(R(np.pi/2, - np.pi / 2, 0, 1), 'Rx(-π)12')
    u2 = U(R(0, np.pi, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, np.pi, 0, 1), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(-π/2)01')
    xx = rTwoQuditMSGate01()
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)

def rCX_clear02(cirquit, q1, q2):
    u1 = U(R(np.pi / 2, - np.pi / 2, 0, 1), 'Rx(-π)12')
    u2 = U(R(0, np.pi, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, np.pi, 0, 2), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(-π/2)01')
    xx = rTwoQuditMSGate02()
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)

class H(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi / 2, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'U_enc'

class X1_conj(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, complex(0, -1), 0], [complex(0, -1), 0, 0], [0, 0, 1]])

    def _circuit_diagram_info_(self, args):
        return 'X1_c'

class X2_conj(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.conj(np.array([[0, 0, complex(0, -1)],
                                 [0, 1, 0],
                                 [complex(0, -1), 0, 0]]))

    def _circuit_diagram_info_(self, args):
        return 'X2_c'

class Z1(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'Z1'

class Y1(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(np.pi / 2, np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'Y1'

class X12(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 1, 2)

    def _circuit_diagram_info_(self, args):
        return 'X2'

class X12r(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, -np.pi, 1, 2)

    def _circuit_diagram_info_(self, args):
        return 'X2'

class X02(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 2)

    def _circuit_diagram_info_(self, args):
        return 'X2'

class X02r(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, -np.pi, 0, 2)

    def _circuit_diagram_info_(self, args):
        return 'X2'


class X1(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'X1'

class X1r(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, -np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'X1'


class Hr(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(np.pi / 2, -np.pi / 2, 0, 1) @ R(0, -np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'U_enc'



def encoding(cirquit, mask, q_mask, v):

    for o in range(B):
        u1 = U(R(v[0 + o * 12], v[1 + o * 12], mask[o][0], mask[o][1]), 'Rx(-π)12')
        u2 = U(R(v[2 + o * 12], v[3 + o * 12], mask[o][0], mask[o][1]), 'Ry(π/2)01')
        u3 = U(R(v[4 + o * 12], v[5 + o * 12], mask[o][0], mask[o][1]), 'Rx(-π)01')
        u4 = U(R(v[6 + o * 12], v[7 + o * 12], mask[o][2], mask[o][3]), 'Ry(-π/2)01')
        u5 = U(R(v[8 + o * 12], v[9 + o * 12], mask[o][2], mask[o][3]), 'Rx(-π)01')
        u6 = U(R(v[10 + o * 12], v[11 + o * 12], mask[o][2], mask[o][3]), 'Ry(-π/2)01')
        xx = TwoQS([mask[o][0], mask[o][1], mask[o][2], mask[o][3]])
        cirquit.append([u1(q_mask[o][0]), u4(q_mask[o][1])], strategy=InsertStrategy.INLINE)
        cirquit.append([xx(q_mask[o][0], q_mask[o][1])], strategy=InsertStrategy.INLINE)
        cirquit.append([u2(q_mask[o][0]), u5(q_mask[o][1])], strategy=InsertStrategy.INLINE)
        cirquit.append([u3(q_mask[o][0]), u6(q_mask[o][1])], strategy=InsertStrategy.INLINE)

        #([np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0])

def decoding(cirquit, mask, q_mask, v):
    for o in range(B-1,-1,-1):

        u1 = U(R(v[0 + o * 12], -v[1 + o * 12], mask[o][0], mask[o][1]), 'Rx(-π)12')
        u2 = U(R(v[2 + o * 12], -v[3 + o * 12], mask[o][0], mask[o][1]), 'Ry(π/2)01')
        u3 = U(R(v[4 + o * 12], -v[5 + o * 12], mask[o][0], mask[o][1]), 'Rx(-π)01')
        u4 = U(R(v[6 + o * 12], -v[7 + o * 12], mask[o][2], mask[o][3]), 'Ry(-π/2)01')
        u5 = U(R(v[8 + o * 12], -v[9 + o * 12], mask[o][2], mask[o][3]), 'Rx(-π)01')
        u6 = U(R(v[10 + o * 12], -v[11 + o * 12], mask[o][2], mask[o][3]), 'Ry(-π/2)01')
        xx = rTwoQS([mask[o][0], mask[o][1], mask[o][2], mask[o][3]])
        cirquit.append([u3(q_mask[o][0]), u6(q_mask[o][1])], strategy=InsertStrategy.INLINE)
        cirquit.append([u2(q_mask[o][0]), u5(q_mask[o][1])], strategy=InsertStrategy.INLINE)
        cirquit.append([xx(q_mask[o][0], q_mask[o][1])], strategy=InsertStrategy.INLINE)
        cirquit.append([u1(q_mask[o][0]), u4(q_mask[o][1])], strategy=InsertStrategy.INLINE)

zZ = np.array([[1,0,0]]).T
eE = np.array([[0,1,0]]).T
fF = np.array([[0,0,1]]).T
A = [zZ, eE, fF]

B = []

def m(a ,b, c, d, e):
    return np.kron(np.kron(np.kron(np.kron(a, b), c), d), e)

for i1 in range(3):
    for i2 in range(3):
        B.append(np.kron(A[i1], A[i2]))

x01 = X1()
x12 = X12()
x02 = X02()
hr = Hr()
h = H()

def operation(v):
    vsp = []
    for i in range(B):
        vsp.append(v)
    vsp = np.array(vsp)
    v = np.reshape(vsp, (1, B * 12))[0]
    circuit1 = cirq.Circuit()
    qutrits1 = []
    qutrits1.append(cirq.LineQid(0, dimension=3))
    qutrits1.append(cirq.LineQid(1, dimension=3))
    qutrits1.append(cirq.LineQid(2, dimension=3))
    qutrits1.append(cirq.LineQid(3, dimension=3))

    q1, q2, q3= qutrits1[0], qutrits1[1], qutrits1[2]
    q4 = qutrits1[3]

    Q_mask = [(q1,q3), (q3,q1), (q2,q3), (q3,q2), (q1,q2)]
    Q_mask = [(q1, q3), (q3, q1), (q2, q3), (q3, q2), (q1, q2), (q2, q1)]

    mask = [(0,2,0,2), (0,1,0,1), (0,2,1,2),(0,1,0,1), (0,2,0,2)]
    mask = [(0, 2, 0, 2), (0, 1, 0, 1), (0, 2, 1, 2), (0, 1, 0, 1), (0, 2, 0, 2), (0, 1, 0, 1)]

    alf11 = random.randint(0, 1000) / 1000 * 2 * np.pi
    alf21 = random.randint(0, 1000) / 1000 * 2 * np.pi
    povorot = R(alf11, alf21, 0, 1)
    pg = U(povorot)
    circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)

    alf12 = random.randint(0, 1000) / 1000 * 2 * np.pi
    alf22 = random.randint(0, 1000) / 1000 * 2 * np.pi
    povorot = R(alf12, alf22, 0, 1)
    pg = U(povorot)
    circuit1.append([pg(qutrits1[1])], strategy=InsertStrategy.INLINE)

    alf13 = random.randint(0, 1000) / 1000 * 2 * np.pi
    alf23 = random.randint(0, 1000) / 1000 * 2 * np.pi
    povorot = R(alf13, alf23, 0, 1)
    pg = U(povorot)
    circuit1.append([pg(qutrits1[2])], strategy=InsertStrategy.INLINE)

    encoding(circuit1, mask, Q_mask, v)

    Q_mask = [(q4, q3), (q3, q4), (q2, q3), (q3,q2), (q4, q2), (q2,q4)]

    decoding(circuit1, mask, Q_mask, v)

    povorot = R(alf11, -alf21, 0, 1)
    pg = U(povorot)
    circuit1.append([pg(qutrits1[3])], strategy=InsertStrategy.INLINE)

    povorot = R(alf12, -alf22, 0, 1)
    pg = U(povorot)
    circuit1.append([pg(qutrits1[1])], strategy=InsertStrategy.INLINE)

    povorot = R(alf13, -alf23, 0, 1)
    pg = U(povorot)
    circuit1.append([pg(qutrits1[2])], strategy=InsertStrategy.INLINE)
    #circuit1.append([cirq.measure(qutrits1[0])])
    #circuit1.append([cirq.measure(qutrits1[1])])
    #circuit1.append([cirq.measure(qutrits1[2])])
    #sim = cirq.Simulator()
    #res = sim.simulate(circuit1)
    ro_ab = cirq.final_state_vector(circuit1, qubit_order=qutrits1)

    '''
    measured_bit = res.measurements[str(qutrits1[0])][0]
    print(f'Measured bit: {measured_bit}')
    measured_bit = res.measurements[str(qutrits1[1])][0]
    print(f'Measured bit: {measured_bit}')
    measured_bit = res.measurements[str(qutrits1[2])][0]
    print(f'Measured bit: {measured_bit}')
    '''
    #ro_ab = 1
    return ro_ab


B = 6
'''

guess =np.array([np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0])
delta = np.pi
for i in range(72):
    guess[i] = guess[i] + random.randint(-1000, 1000) / 1000 * delta

guess = np.array(guess)
print(operation(guess))
'''
'''
adelta = np.linspace(0,np.pi,10)

guess = np.array(guess)
y = []
for delta in adelta:
    print(delta)
    guess = np.array(
        [np.pi / 2, np.pi / 2, 0, -np.pi, np.pi / 2, -np.pi / 2, 0, 0, 0, -np.pi, 0, 0, np.pi / 2, np.pi / 2, 0, -np.pi,
         np.pi / 2, -np.pi / 2, 0, 0, 0, -np.pi, 0, 0, np.pi / 2, np.pi / 2, 0, -np.pi, np.pi / 2, -np.pi / 2, 0, 0, 0,
         -np.pi, 0, 0, np.pi / 2, np.pi / 2, 0, -np.pi, np.pi / 2, -np.pi / 2, 0, 0, 0, -np.pi, 0, 0, np.pi / 2,
         np.pi / 2, 0, -np.pi, np.pi / 2, -np.pi / 2, 0, 0, 0, -np.pi, 0, 0, np.pi / 2, np.pi / 2, 0, -np.pi, np.pi / 2,
         -np.pi / 2, 0, 0, 0, -np.pi, 0, 0])

    for i in range(72):

        guess[i] = guess[i] + random.randint(-1000, 1000) / 1000 * delta
    #guess.append(0.2)
    guess = np.array(guess)
    slg = 0
    for j in range(20):
        slg += operation(guess) / 20
    y.append(slg)



bnds = []
for i in range(72):
    bnds.append((-np.pi, np.pi))

res1 = scipy.optimize.minimize(operation, guess, method= 'COBYLA', bounds=bnds)
print(res1)
print(list(res1.x))
'''

guess =np.array([1.8661683312064359, 0.8725141878731524, 2.603527507827626, -1.563611811899032, 3.0843550051242703, -1.4145873438076917, 3.1177649933496396, 2.53579340834936, -2.5590059466078774, -2.2006950844016973, 0.06129517714744294, 1.3378418752741235, 1.0423248794958209, -0.7978972080383651, 3.128035134364833, -1.2981962702990164, 2.2021542047151916, 1.048630383329516, 1.3962943329854731, -2.408125063969887, -1.5701709458023911, -1.8744624419162026, -1.2431409421244488, -1.4227600616788736, -0.8333294248831615, 3.0220400633341495, -1.1879217700464015, 0.15032490315227395, 2.4626948417833474, 2.1353455067665412, 2.2251880253908243, 1.0236074582776462, 0.5775521906707249, -1.3396830912909703, 0.8690256851190591, -0.21867725446232184, 1.018678494468022, -0.291518320589253, -0.9774680921583486, -3.0573640740835266, 0.5586983144338382, 1.2523925749218443, 1.932742902037953, -1.013579133146362, -2.0186393718309454, -3.093978268648245, 0.41054922428760715, 0.4734444891559942, 3.14054344299952, -0.9612919560531532, -3.141592653589794, -1.3439054111139037, -1.261860270575594, 1.3239552654655422, 1.796205904622592, 0.24856463469470222, -1.1634532694919397, -2.9047349212259337, 2.9641945631978075, 0.7421815479736953])
guess = np.array([0.6467555034813697, 3.1415926535897953, -1.0674154344400453, -3.1368745975635797, 2.0396368987928155, 1.373958559895677, -1.5330859255357512, 1.6866645527124728, -1.6137345708405715, -1.7291988222249353, 0.9372241914210453, 2.903054093146056, 1.8051981692135908, -1.4253848052791371, -2.5708334571124616, -0.24555682626460135, 1.2499600299150624, -1.6307598523687117, -1.5144303841770939, 1.4044492948223426, -2.9860605156838407, -0.683560193942837, 1.4747389813669673, 1.4540929128074873, -0.031244252651775443, 2.3129761759100758, 0.8458212002847437, -2.7755592613023703, 2.127262098033904, 0.6194600386157647, 0.5608664403350369, -1.8626230726821524, -1.0460621527521108, -3.1415926535897953, 0.49235055779571396, -1.744700994058268, 3.1102313335374623, 3.0134845961252856, 2.635309031972392, -2.71235651363571, 2.9366656949378838, -2.4601487626897267, 0.7370239661660853, 1.7880828738770806, 2.607140792374427, -1.3904435835279643, -0.8945028430378297, 3.1415926535897962, 2.7080827337589755, -2.7254075928105226, -0.8145418923116667, -2.5318740918297253, 2.440495191188439, -0.9238951839543758, -1.3137659999152962, -0.19783125391424766, 1.4133214703002945, -2.393330025145953, -2.1421964003427933, 2.2880713979468887, 1.2547284817293747, 2.873735932411793, -1.2714418139741643, -2.4560419971583705, -1.067160620621314, 1.6356710830655206, -1.63989810253236, 1.6128883428401464, -3.1415926535897953, -2.9444578562634725, 1.4808706593197605, 1.6354547007730602])
guess = np.array([-1.5711481228664812, 1.5709276753943524, -2.055122750214922, -1.9908447933791968, 0.17558477087655475, -0.6296411307822198, -0.00032083835775867717, 1.9375485039540794, 0.08901630501248764, -1.9445230412641006, -1.543020491533192, 3.058103826825693, 2.308179041573644, 3.116481179941959, 1.4408664280208727, 0.06590132914946709, 1.6807318022039233, -2.6661778092728725, 2.696338985208746, 0.0002383452155620684, 0.4325281522505938, -3.1406649325351923, -0.7452110632785461, 3.140959439285155, 1.7603463223483924, 0.7712034740176439, 3.054066469617223, -2.8247879289221207, 1.6491420533793222, -0.26454157895465447, 2.3562666539117743, 0.19325649230607253, 2.533896214786839, -3.1330955702262715, -1.323734299096848, 1.9777039912352625, 2.6331493496929776, 2.6351011018254624, 1.3323562685997237, -1.0681953104513375, 3.0953291481679717, 1.121142279497629, 3.0379728345080244, 3.13948193212777, -3.0981282701206, -2.4211457634710256, -2.9237248235504647, -1.0778547472816262, 3.120713195392774, -1.1977379904273235, 0.15858131988477603, 0.32877984783003894, 0.32828281399481557, -0.2844302505860814, 1.382990763954274, -1.9721497159117303, -2.3728315497103645, -1.1841135767347621, 2.62371268598078, -0.325925368249746, 2.890800302071034, 0.9171783200804831, -0.10435452985186555, -3.14010726471783, 1.3866245465370577, -1.5679265783758474, 1.5714004009148264, -1.5698834700465107, -2.718166621901453, -0.7689388614532009, 1.6804136994490828, -1.860692641100222])
guess = np.array([0.21204632009689167, 3.0232604246108052, -1.1936155556579255, -2.6010233589327454, 2.1437527859957077, -2.6479433842683067, -2.935669811035543, 3.116633086611407, 2.6782738240620003, -3.1164532938052005, 0.7342955012067627, -3.125706456643734, 3.139571518831329, 2.134534065390125, 1.506348048023131, -1.9845285403743222, -0.6053856635387032, -1.7409895619168887, 2.0478932514429373, 3.1415399957087113, 0.6784644543469778, -3.1409811609198717, 2.2663538973128143, -0.0009040705762414594, 1.7575562273267011, 3.1415535132096335, -2.82854870589543, -3.14157950512321, 3.13362178742995, -3.141592653589793, 1.5698008282462401, 1.570585811153751, 2.361881903972472, 0.26148677978775514, -1.5887697390057427, 1.7533619254437312, 2.896772921813066, 3.1412024881140503, 0.8104127428307758, -3.141528621296489, 0.8657883604583537, -3.141416286542616, 1.844131842973893, -0.2808428363466231, 3.063110359451788, 0.016217060316392883, -1.7876779978243156, -2.864136640092026, 2.20127710530967, 3.099504227715247, 0.9517347675059561, 0.007373202044742346, 2.717894662922935, -3.102697976536852, -0.07039479800074062, 2.9766443687647093, -1.173984931563637, -1.6885974959494658, 2.125423173973723, 1.3916090468105824, 3.1410865137038995, 1.5411663376982896, -0.7204641599677238, -1.8016981518035602, 1.066229924367594, 2.40114445110796, -1.2240083689866605, 3.0602941961077743, 1.2958565214368196, -2.684609623720595, 1.2761836433231426, 2.765359517516925])
guess = np.array([-1.5369901329429743, 3.0124193643061323, 0.5259941132102801, -2.6087974931265343, 2.947940695493949, -2.6373615067329435, 1.5767023737337487, -1.569060769764874, 1.6958058091676236, 1.6632933659831737, -0.578339567643413, 0.14838831584315068, -2.747222537316023, -1.8514475542015472, -3.1323547216737198, 0.6902810956820167, 2.7628091553272935, -1.9436386629211027, -0.39483842754213766, 0.16342686951977475, -1.669503817716026, 1.6347596095087618, 1.6145009880633812, -1.5856328455186868, -0.021133104290521208, 1.3043289354828647, 3.1415926535897922, 1.171963303249103, 2.7451015578019726, 0.1313311848534879, 1.0717420972207548, 0.05615569926385866, 3.1090798209796917, 0.08503544915675075, -0.5923621167085412, -3.0707074666464544, -0.3038903565897487, 2.6864305240158464, 0.5557088375835312, 0.6111685821474183, -2.006802015489492, 0.20518343741228823, -0.5116865142954414, 0.568780036067177, 0.5391759954773385, 3.0186906800947866, 0.566975655874993, -0.4448039359802568, -1.1782428689660964, -1.63810789645773, -1.4567233510756394, 0.6200930859914643, -1.266165770283553, 1.2221044106351984, 0.6110760463873959, 3.1228266474656055, 1.5343820217331583, 0.020716529973795125, 2.3178732264110113, -0.041646202896222624, 3.141592653589795, 1.9998886042880235, 0.7840072326308802, -3.081198376973686, -1.5965311033606624, -1.9544351722359252, 0.30622685272067907, 0.8947123684096949, 2.7921022169250516, -1.9222953682973047, -0.23304754267898736, 0.3180781790777861])
guess = np.array([3.07053502,  0.05762438,  0.07586133 ,-3.14159265,  3.14062118, -0.05712138, -0.20885243, -0.04771057, -0.79354331 ,-2.02178975 , 1.21798736 ,-1.91873761])
guess =np.array([np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0])
#print(guess[0:12] - guess[2 * 12:3 * 12])
guess = guess[0:12]


print(len(guess))
print(operation(guess))