import pytest

from mentpy.mbqc.mbqcircuit import MBQCircuit
from mentpy.mbqc import flow
import mentpy as mp

# Test cases
# Taken from Generalized Flow paper arxiv:0702212.pdf

@pytest.fixture
def test_circuit(request):
     yield request.param()

def gs_cf():
    """
    A graph with flow. (pg. 12 Generalized Flow and Determinism)
    - g(a) = d
    - g(b) = e
    - g(c) = f
    - Let a = 0, b = 2, c = 4, d = 1, e = 3, f = 5
    """
    gs_gf_nf = mp.GraphState()
    gs_gf_nf.add_edges_from(([(0, 1), (2, 3), (4, 5)]))
    return mp.MBQCircuit(gs_gf_nf, input_nodes=[0, 2, 4], output_nodes=[1, 3, 5])


def gs_gf_ncf():
    """
    A graph with generalized flow but no flow. (pg. 11 Generalized Flow and Determinism)
    - g(a) = d
    - g(b) = e
    - g(c) = {d, f}
    - Let a = 0, b = 2, c = 4, d = 1, e = 3, f = 5
    """
    gs_gf_nf = mp.GraphState()
    gs_gf_nf.add_edges_from(([(0, 1), (2, 3), (4, 5), (4, 1)]))
    return mp.MBQCircuit(gs_gf_nf, input_nodes=[0, 2, 4], output_nodes=[1, 3, 5])


def ogs_ngf_pf():
    """"
    Open graph state having no generalized flow but has a pauli flow (pg 14 Generalized Flow and Determinism)
    Labels of the circuit from the Generalized Flow and Determinism paper are offset by one.
    """
    og_gf = mp.GraphState()
    og_gf.add_edges_from(
        [
            (0, 2),
            (2, 5),
            (5, 8),
            (8, 10),
            (2, 3),
            (3, 4),
            (3, 6),
            (1, 4),
            (4, 7),
            (7, 9),
            (9, 11),
            (6, 7),
            (5, 6),
        ]
    )
    return mp.MBQCircuit(og_gf, input_nodes=[0, 1], output_nodes=[10, 11])


def ogs_ngf_npf():
    """
    Open graph state with no Pauli or generalized flow implementing a deterministic pattern for the swap operator
    Labels of the circuit from the Generalized Flow and Determinism paper are offset by one.
    """
    # NOTE: how to test for determinism without flow?
    ogs_ngf_npf = mp.GraphState()
    ogs_ngf_npf.add_edges_from(
        [
            (0, 2),
            (2, 5),
            (5, 8),
            (8, 11),
            (3, 6),
            (6, 9),
            (1, 4),
            (4, 7),
            (7, 10),
            (10, 12),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (8, 9),
            (9, 10),
        ]
    )
    return mp.MBQCircuit(ogs_ngf_npf, input_nodes=[0, 1], output_nodes=[11, 12])

def ogs_cf5_gf2():
    """
    An open graph state with flow of depth five and a generalized flow of depth 2. (pg. 13 Generalized Flow and Determinism)
    """
    ogs_cf5_gf2 = mp.GraphState()
    ogs_cf5_gf2.add_edges_from(([(0, 1), (2, 3), (4, 5), (1, 6), (3, 7), (5, 8)]))
    return mp.MBQCircuit(ogs_cf5_gf2, input_nodes=[0, 2, 4], output_nodes=[6, 7, 8])

def cnot():
    """
    A CNOT gate.
    Should have pflow but not gflow or cflow.
    """
    controlled_a = 1
    controlled_b = 4
    cnot = mp.GraphState()
    cnot.add_edges_from([(0, 1), (controlled_a, 2), (controlled_a, controlled_b), (3, controlled_b), (4, 5)])
    return mp.MBQCircuit(cnot, input_nodes=[0, 3], output_nodes=[2, 5])


def small_cnot():
    """
    A CNOT gate with a single control qubit.
    Should have pflow but not gflow or cflow, but as the topology is different from the cnot gate, it is worth testing.
    """
    small_cnot = mp.GraphState()
    ctrl = 3
    small_cnot.add_edges_from([(0, 1), (1, 2), (1, ctrl)])
    # ensure that the input and output sets share the control qubit
    return mp.MBQCircuit(small_cnot, input_nodes=[0, ctrl], output_nodes=[2, ctrl])


def ring():
    """
    A ring of 6 qubits.
    Should have causal flow when input and output sets are the same size
    """
    ring = mp.GraphState()
    ring.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    return mp.MBQCircuit(ring, input_nodes=[0, 1], output_nodes=[4, 5])


class TestFlow:
    CFLOW, GFLOW, PFLOW = True, True, False

    # TODO: add tests for the following:

    # - cflow but no gflow or pflow
    # - gflow but no cflow or pflow
    # - pflow but no cflow or gflow
    # - cflow and pflow but no gflow
    # - gflow and pflow but no cflow
    # - cflow and gflow but no pflow
    # - cflow, gflow, and pflow
    # - no cflow, gflow, or pflow
    # - no cflow, gflow, or pflow but is deterministic
    # - no cflow, gflow, or pflow and is not deterministic

    # NOTE: "None" is used to indicate that we don't care about the value of the flow or don't know it.
    @pytest.mark.parametrize(
        "name, test_circuit, expect_cflow, expect_gflow, expect_pflow",
        [
            ("Graph state with causal flow but no general flow", gs_cf, True, None, None),
            ("Graph state with general flow but no causal flow", gs_gf_ncf, False, True, None),
            ("Open graph state with causal flow and general flow", ogs_cf5_gf2, True, True, None),
            ("Open graph state with Pauli flow, but no general flow", ogs_ngf_pf, None, False, True),
            ("Open graph state with no general flow and no Pauli flow, but is deterministic", ogs_ngf_npf, False, False, False),
            ("CNOT gate", cnot, False, False, True),
            ("Small CNOT gate", small_cnot, None, None, True),
            ("Ring of 6 Qubits", ring, True, True, True),
        ],
        indirect=["test_circuit"]
    )
    def test_flows(self, name: str, test_circuit: MBQCircuit, expect_cflow: bool, expect_gflow: bool, expect_pflow: bool):

        # TODO: add test for deterministic circuits with no flow
        # HACK: generators might not be the best way to do this
        has_cflow = flow.find_cflow(
                test_circuit.graph,
                test_circuit.input_nodes,
                test_circuit.output_nodes,
            )[0] if self.CFLOW else None

        has_gflow = flow.find_gflow(
                test_circuit.graph,
                test_circuit.input_nodes,
                test_circuit.output_nodes
            )[0] if self.GFLOW else None

        has_pflow = flow.find_pflow(
                test_circuit.graph,
                test_circuit.input_nodes,
                test_circuit.output_nodes,
            )[0] if self.PFLOW else None

        print(f"{name}: has cflow: {has_cflow}; has gflow: {has_gflow}; has pflow: {has_pflow}")

        if self.CFLOW and not expect_cflow is None: assert has_cflow is expect_cflow
        if self.GFLOW and not expect_gflow is None: assert has_gflow is expect_gflow
        if self.PFLOW and not expect_pglow is None: assert has_pflow is expect_pflow
