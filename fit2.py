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
    for o in range(6):
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
    for o in range(5,-1,-1):
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
    #print(1)
    um = 0
    B = 6
    for _ in range(3):
        circuit1 = cirq.Circuit()
        qutrits1 = []
        qutrits1.append(cirq.LineQid(0, dimension=3))
        qutrits1.append(cirq.LineQid(1, dimension=3))
        qutrits1.append(cirq.LineQid(2, dimension=3))
        qutrits1.append(cirq.LineQid(3, dimension=3))

        q1, q2, q3 = qutrits1[0], qutrits1[1], qutrits1[2]
        q4 = qutrits1[3]

        Q_mask = [(q1, q3), (q3, q1), (q2, q3), (q3, q2), (q1, q2), (q2, q1)]
        mask = [(0, 2, 0, 2), (0, 1, 0, 1), (0, 2, 1, 2), (0, 1, 0, 1), (0, 2, 0, 2), (0, 1, 0, 1)]

        # cмена нач сост:
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

        Q_mask = [(q4, q3), (q3, q4), (q2, q3), (q3, q2), (q4, q2), (q2, q4)]

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

        ro_ab = cirq.final_density_matrix(circuit1, qubit_order=qutrits1)
        um  = um + abs(ro_ab[0][0] - 1)

    return um/3


def operation2(v):
    #print(1)
    um = 0
    B = 6
    A = [0, 1]
    for i in A:
        circuit1 = cirq.Circuit()
        qutrits1 = []
        qutrits1.append(cirq.LineQid(0, dimension=3))
        qutrits1.append(cirq.LineQid(1, dimension=3))
        qutrits1.append(cirq.LineQid(2, dimension=3))
        qutrits1.append(cirq.LineQid(3, dimension=3))

        q1, q2, q3 = qutrits1[0], qutrits1[1], qutrits1[2]
        q4 = qutrits1[3]

        Q_mask = [(q1, q3), (q3, q1), (q2, q3), (q3, q2), (q1, q2), (q2, q1)]
        mask = [(0, 2, 0, 2), (0, 1, 0, 1), (0, 2, 1, 2), (0, 1, 0, 1), (0, 2, 0, 2), (0, 1, 0, 1)]

        # cмена нач сост:
        circuit1.append([x01(qutrits1[i])], strategy=InsertStrategy.INLINE)

        encoding(circuit1, mask, Q_mask, v)

        Q_mask = [(q4, q3), (q3, q4), (q2, q3), (q3, q2), (q4, q2), (q2, q4)]

        decoding(circuit1, mask, Q_mask, v)

        if

        circuit1.append([x01(qutrits1[i])], strategy=InsertStrategy.INLINE)

        um  = um + abs(ro_ab[0][0] - 1)

    return um/3






guess =np.array([np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0,np.pi / 2, np.pi / 2,0,-np.pi, np.pi / 2, -np.pi / 2, 0, 0,0, -np.pi, 0, 0])
delta = np.pi
for i in range(72):
    guess[i] = guess[i] + random.randint(-1000, 1000) / 1000 * delta

guess = np.array(guess)
print(operation(guess))

bnds = []
for i in range(72):
    bnds.append((-np.pi, np.pi))

res1 = scipy.optimize.minimize(operation, guess, method= 'COBYLA', bounds=bnds)
res2 = scipy.optimize.minimize(operation, res1.x, method= 'COBYLA', bounds=bnds)
print(res1)
print(list(res1.x))
