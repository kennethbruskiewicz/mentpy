# Author: Luis Mantilla, Kenneth Bruskiewicz
# Github: BestQuark
"""This is the Flow module. It deals with the flow of a given graph state"""
import math
import numpy as np
import networkx as nx

from typing import Generator, List, Union, Callable, Iterator, Tuple, Dict, Any
from functools import partial
from itertools import product
import warnings

import galois
from mentpy.operators import Ment
from mentpy.mbqc import GraphState

FlowResult = Tuple[bool, Callable | None, Callable | None, int | None]  # TODO: tougher typing constraints
Resource = Union[GraphState, None]                 # TODO: Add MBQCircuit after resolving circular import

## Not used in main MBQC module
def _flow_from_array(graph: GraphState, input_nodes, output_nodes, f: List):
    """Create a flow function from a given array f"""

    def flow(v):
        if v in [v for v in graph.nodes() if v not in output_nodes]:
            return int(f[v])
        else:
            raise UserWarning(f"The node {v} is not in domain of the flow.")

    return flow


def _get_chain_decomposition(
    graph: GraphState, input_nodes, output_nodes, C: nx.DiGraph
):
    """Gets the chain decomposition"""
    P = np.zeros(len(graph))
    L = np.zeros(len(graph))
    f = {v: 0 for v in set(graph) - set(output_nodes)}
    for i in input_nodes:
        v, l = i, 0
        while v not in output_nodes:
            f[v] = int(next(C.successors(v)))
            P[v] = i
            L[v] = l
            v = int(f[v])
            l += 1
        P[v], L[v] = i, l
    return (f, P, L)


def _compute_suprema(graph: GraphState, input_nodes, output_nodes, f, P, L):
    """Compute suprema

    status: 0 if none, 1 if pending, 2 if fixed.
    """
    (sup, status) = _init_status(graph, input_nodes, output_nodes, P, L)
    for v in set(graph.nodes()) - set(output_nodes):
        if status[v] == 0:
            (sup, status) = _traverse_infl_walk(
                graph, input_nodes, output_nodes, f, sup, status, v
            )

        if status[v] == 1:
            return None

    return sup


def _traverse_infl_walk(
    graph: GraphState, input_nodes, output_nodes, f, sup, status, v
):
    """Compute the suprema by traversing influencing walks

    status: 0 if none, 1 if pending, 2 if fixed.
    """
    status[v] = 1
    vertex2index = {v: index for index, v in enumerate(input_nodes)}

    for w in list(graph.neighbors(f[v])) + [f[v]]:
        if w != v:
            if status[w] == 0:
                (sup, status) = _traverse_infl_walk(
                    graph, input_nodes, output_nodes, f, sup, status, w
                )
            if status[w] == 1:
                return (sup, status)
            else:
                for i in input_nodes:
                    if sup[vertex2index[i], v] > sup[vertex2index[i], w]:
                        sup[vertex2index[i], v] = sup[vertex2index[i], w]
    status[v] = 2
    return sup, status


def _init_status(graph: GraphState, input_nodes: List, output_nodes: List, P, L):
    """Initialize the supremum function

    status: 0 if none, 1 if pending, 2 if fixed.
    """
    sup = np.zeros((len(input_nodes), len(graph.nodes())))
    vertex2index = {v: index for index, v in enumerate(input_nodes)}
    status = np.zeros(len(graph.nodes()))
    for v in graph.nodes():
        for i in input_nodes:
            if i == P[v]:
                sup[vertex2index[i], v] = L[v]
            else:
                sup[vertex2index[i], v] = len(graph.nodes())

        status[v] = 2 if v in output_nodes else 0

    return sup, status


def _build_path_cover(graph: GraphState, input_nodes: List, output_nodes: List):
    """Builds a path cover

    Given a directed graph G = (V, E):
        The path cover is a set of directed paths such that _every_ vertex v \el V belongs to at least one path.
        A path cover may include paths of length zero (one vertex).

    status: 0 if 'fail', 1 if 'success'
    """
    fam = nx.DiGraph()
    visited = np.zeros(graph.number_of_nodes())
    iter = 0
    for i in input_nodes:
        iter += 1
        (fam, visited, status) = _augmented_search(
            graph, input_nodes, output_nodes, fam, iter, visited, i
        )
        if not status:
            return status

    if not len(set(graph.nodes) - set(fam.nodes())):
        return fam

    return 0


def _augmented_search(
    graph: GraphState,
    input_nodes: List,
    output_nodes: List,
    fam: nx.DiGraph,
    iter: int,
    visited,
    v,
):
    """Does an augmented search

    status: 0 if 'fail', 1 if 'success'
    """
    visited[v] = iter
    if v in output_nodes:
        return (fam, visited, 1)
    if (
        (v in fam.nodes())
        and (v not in input_nodes)
        and (visited[next(fam.predecessors(v))] < iter)
    ):
        fam, visited, status = _augmented_search(
            graph,
            input_nodes,
            output_nodes,
            fam,
            iter,
            visited,
            next(fam.predecessors(v)),
        )
        if status:
            # Remove edge doesn't have a return type, it removes the edge in place
            fam.remove_edge(next(fam.predecessors(v)), v)
            return fam, visited, 1

    for w in graph.neighbors(v):
        if (visited[w] < iter) and (w not in input_nodes) and (not fam.has_edge(v, w)):
            if w not in fam.nodes():
                fam, visited, status = _augmented_search(
                    graph, input_nodes, output_nodes, fam, iter, visited, w
                )
                if status:
                    fam.add_edge(v, w)
                    return (fam, visited, 1)
            elif visited[next(fam.predecessors(w))] < iter:
                fam, visited, status = _augmented_search(
                    graph,
                    input_nodes,
                    output_nodes,
                    fam,
                    iter,
                    visited,
                    next(fam.predecessors(w)),
                )
                if status:
                    fam.remove_edge(next(fam.predecessors(w)), w)
                    fam.add_edge(v, w)
                    return fam, visited, 1

    return fam, visited, 0

def _find_flow(graph: GraphState, input_nodes, output_nodes, sanity_check=True) -> object:
    """Finds the generalized flow of graph state if allowed.

    Implementation of https://arxiv.org/pdf/quant-ph/0603072.pdf.

    Returns
    -------
    The flow function ``flow`` and the partial order function.

    Group
    -----
    states
    """
    # raise deprecated warning
    warnings.warn(
        "The function find_flow is deprecated. Use find_cflow instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Helper functions
    # ----------------
    # check if a partial order is a valid partial order
    int_partial_order = lambda sigma, x, y: sigma[vertex2index[int(P[y])], int(x)] <= L[y]

    n_input, n_output = len(input_nodes), len(output_nodes)
    inp = input_nodes
    outp = output_nodes
    if n_input != n_output:
        raise ValueError(
            f"Cannot find flow or gflow. Input ({n_input}) and output ({n_output}) nodes have different size."
        )

    initial_graph = GraphState(graph.copy())

    # create a copy of the object state
    update_labels = False
    # check if labels of graph are integers going from 0 to n-1 and if not, create a mapping
    if not all([i in graph.nodes for i in range(len(graph))]):
        mapping = {v: i for i, v in enumerate(graph.nodes)}
        inverse_mapping = {i: v for i, v in enumerate(graph.nodes)}
        new_graph = nx.relabel_nodes(initial_graph.copy(), mapping)
        inp, outp = [mapping[v] for v in input_nodes], [
            mapping[v] for v in output_nodes
        ]
        graph = GraphState(new_graph)
        update_labels = True

    tau = _build_path_cover(graph, inp, outp)
    if tau:
        f, P, L = _get_chain_decomposition(graph, inp, outp, tau)
        sigma = _compute_suprema(graph, inp, outp, f, P, L)

        if sigma is not None:
            int_flow = _flow_from_array(graph, inp, outp, f)
            vertex2index = {v: index for index, v in enumerate(inp)}

            # if labels were updated, update them back
            if update_labels:
                graph = initial_graph
                flow = lambda v: inverse_mapping[int_flow(mapping[v])]
                partial_order = lambda x, y: int_partial_order(sigma, mapping[x], mapping[y])
            else:
                flow = int_flow
                partial_order = int_partial_order

            state_flow = (flow, partial_order)
            if sanity_check:
                if not check_if_flow(graph, inp, outp, flow, partial_order):
                    raise RuntimeError(
                        "Sanity check found that flow does not satisfy flow conditions."
                    )
            return state_flow

        else:
            warnings.warn(
                "The given state does not have a flow.", UserWarning, stacklevel=2
            )
            return None, None
    else:
        warnings.warn(
            "Could not find a flow for the given state.", UserWarning, stacklevel=2
        )
        return None, None


### This section implements causal flow

def _find_cflow(graph: GraphState, input_nodes, output_nodes) -> FlowResult:
    """Finds the causal flow of a ``MBQCGraph`` if it exists.
    Retrieved from https://arxiv.org/pdf/0709.2670v1.pdf.
    """
    if len(input_nodes) != len(output_nodes):
        raise ValueError(
            f"Cannot find flow or gflow. Input ({len(input_nodes)}) and output ({len(output_nodes)}) nodes have different size."
        )

    l = {}
    g = {}
    past = {}
    C_set = set()

    graph_extended = graph.copy()
    max_node = max(graph.nodes()) + 1
    input_nodes_extended = [max_node + i for i in range(len(input_nodes))]
    graph_extended.add_edges_from(
        [(input_nodes_extended[i], input_nodes[i]) for i in range(len(input_nodes))]
    )

    for v in graph_extended.nodes():
        l[v] = 0
        past[v] = 0

    for v in set(output_nodes) - set(input_nodes_extended):
        past[v] = len(
            set(graph_extended.neighbors(v))
            & (set(graph_extended.nodes() - set(output_nodes)))
        )
        if past[v] == 1:
            C_set = C_set.union({v})

    flow, l = causal_flow_aux(
        graph_extended,
        set(input_nodes_extended),
        set(output_nodes),
        C_set,
        past,
        1,
        g,
        l,
    )

    flow = {k: v for k, v in flow.items() if k not in input_nodes_extended}
    ln = {k: v for k, v in l.items() if k not in input_nodes_extended}

    if len(flow) != len(graph.nodes()) - len(output_nodes):
        return False, None, None, None

    return True, lambda x: flow[x], lambda u, v: ln[u] > ln[v], max(flow.values())


def causal_flow_aux(graph: nx.Graph, inputs, outputs, C, past, k, g, l) -> Tuple[Dict, Dict]:
    """Aux function for causal_flow"""
    V = set(graph.nodes())
    C_prime = set()

    for _, v in enumerate(C):
        # get intersection of neighbors of v and (V \ output nodes
        intersection = set(graph.neighbors(v)) & (V - outputs)
        if len(intersection) == 1:
            u = intersection.pop()
            g[u] = v
            l[u] = k
            outputs.add(u)
            if u not in inputs:
                past[u] = len(set(graph.neighbors(u)) & (V - outputs))
                if past[u] == 1:
                    C_prime.add(u)
            for w in set(graph.neighbors(u)):
                if past[w] > 0:
                    past[w] -= 1
                    if past[w] == 1:
                        C_prime.add(w)

    if len(C_prime) == 0:
        return g, l
    else:
        return causal_flow_aux(graph, inputs, outputs, C_prime, past, k + 1, g, l)

def _find_gflow(graph: GraphState, input_nodes, output_nodes) -> FlowResult:
    """Finds the generalized flow of a ``MBQCGraph`` if it exists.
    Retrieved from https://arxiv.org/pdf/0709.2670v1.pdf.
    """
    graph_extended = GraphState(graph.copy())
    max_node = max(graph.nodes()) + 1
    input_nodes_extended = [max_node + i for i in range(len(input_nodes))]
    graph_extended.add_edges_from(
        [(input_nodes_extended[i], input_nodes[i]) for i in range(len(input_nodes))]
    )

    gamma = nx.adjacency_matrix(graph_extended).toarray()

    l = {}
    g = {}

    for v in output_nodes:
        l[v] = 0

    result, g, l = gflowaux(
        graph_extended,
        gamma,
        set(input_nodes_extended),
        set(output_nodes) - set(input_nodes_extended),
        1,
        g,
        l,
    )

    if result == False:
        warnings.warn("No gflow exists for this graph.", UserWarning, stacklevel=2)
        return False, None, None, None

    gn = {i: g[i] for i in set(graph.nodes()) - set(output_nodes)}
    ln = {i: l[i] for i in graph.nodes()}

    return True, lambda x: gn[x], lambda u, v: ln[u] > ln[v], max(ln.values())


def gf2_matrix_solve(A, b):
    A = A % 2
    b = b % 2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return (sol.round() % 2).astype(int)


def gflowaux(graph: GraphState, gamma, inputs, outputs, k, g, l) -> object:
    """Aux function for gflow"""
    out_prime = set()
    mapping = graph.index_mapping()
    GF = galois.GF(2)
    V = set(graph.nodes())
    C = set()
    vmol = list(V - outputs)
    for u in vmol:
        submatrix = np.zeros((len(vmol), len(outputs - inputs)), dtype=int)
        for i, v in enumerate(vmol):
            for j, w in enumerate(outputs - inputs):
                submatrix[i, j] = gamma[mapping[v], mapping[w]]

        b = np.zeros((len(vmol), 1), dtype=int)
        b[vmol.index(u)] = 1
        solution = gf2_matrix_solve(submatrix, b)

        # Check if solution is a valid solution
        if np.linalg.norm(submatrix @ solution - b) <= 1e-5:
            l[u] = k
            C.add(u)
            g[u] = solution

    if len(C) == 0:
        if set(outputs) == V:
            return True, g, l
        else:
            return False, g, l

    else:
        return gflowaux(graph, gamma, inputs, outputs | C, k + 1, g, l)


def testing_required(testing=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if testing:
                raise NotImplementedError("This algorithm is not yet implemented.")
            return func(*args, **kwargs)

        return wrapper

    return decorator

## This section implements PauliFlow
@testing_required(True)
def _find_pflow(
    graph: GraphState, input_nodes, output_nodes, basis: Union[Ment, str, dict]
) -> FlowResult:
    """Implementation of pauli flow algorithm in https://arxiv.org/pdf/2109.05654v1.pdf"""

    # Order matters here
    if isinstance(basis, Ment):
        basis = basis.plane
    if isinstance(basis, str):
        basis = {v: basis for v in graph.nodes() if v not in input_nodes}
    elif not isinstance(basis, dict):
        raise TypeError("Basis must be a string or a dictionary.")

    lx = set()
    ly = set()
    lz = set()
    A = set()  # set of possible correctors
    B = output_nodes  # set of solved verticies  # TODO why is this output nodes?
    k = 0  # depth
    p = {}  # set of possible correctors at node
    d = {}  # depth at node

    gamma = nx.adjacency_matrix(graph).toarray()

    for v in graph.nodes():
        if v in output_nodes:
            d[v] = 0
        # Python set addition is in-place
        if basis[v] == "X":
            lx.add(v)
        elif basis[v] == "Y":
            ly.add(v)
        elif basis[v] == "Z":
            lz.add(v)

    return pflowaux(graph, gamma, input_nodes, basis, A, B, k, d, p)


def pflowaux(graph: GraphState, gamma, inputs, plane, A, B, k, d, p) -> object:
    """
    Aux function for pflow
    Inputs:
    - graph: the graph state
    - gamma: the adjacency matrix of the graph state
    - inputs: the input nodes
    - plane: the basis of the graph state

    #TODO: Generated by Copilot, check
    - A: the set of nodes that have been measured
    - B: the set of nodes that have not been measured
    - k: the current depth of the algorithm
    - d: the dictionary of delays
    - p: the dictionary of stabilizer generators
    """
    C = set()
    mapping = graph.index_mapping()

    # The following reprsents the primary mechanism for determining if the graph has Pauli flow.
    # The distinguishing feature of Pauli flow is that it is sufficient to show that measurements on
    # the graph state's deterministically produce a Pauli observable on the output nodes, and that this
    # is angle invariant.
    #
    # This is done by checking if the graph state is a stabilizer state, and if so, checking if the stabilizer
    # generators are all Pauli observables on the output nodes. If so, then the graph state has Pauli flow.
    #
    # Following this, similar to gflow and cflow, we attempt to find a solution to the system of equations
    # that delays the measurement of the output nodes. Pauli flow is maximally delayed when a correcting stabilier
    # is found for each output node. This is done by finding a solution to the system of equations that delays the
    # measurement of the output nodes as much as possible.
    #
    # Lemma: If (p, <) is a maximally delayed Pauli flow of (G,I,O,lambda), then (V^<)_0 = {} U V^<XY_0 U V^<XZ_0 U V^<YZ_0

    # primary submatrix
    # common to all planes of measurement to solve for
    # [ Gamma /\ K_Au x P_Au / (Gamma + Id) /\ K_Au x Y_Au ]
    # TODO: mojo

    # Find the witness set K in each plane  # TODO
    # M_A\{mu\} X_K = S\hat{\lambda}
    # X_K is the column vector with 1 in the position of v if v is in K, and 0 otherwise

    # TODO
    # - Ments
    # index in gamma same as index in matrx N x N
    # [] -> output complement
    Lambda_Pu = lambda P, u: {v for v in [] if v != u and v.plane != P}
    Lambda_X = lambda u: partial(Lambda_Pu, P="X")
    Lambda_Y = lambda u: partial(Lambda_Pu, P="X")
    Lambda_Z = lambda u: partial(Lambda_Pu, P="Z")

    # I_C = everything but nodes

    I = lambda _: None

    # The set of possible elements of the witness set K
    K_Au = lambda A, u: (A.union(Lambda_X(u)).union(Lambda_Y(u))).intersection(C(I))
    # The set of verticies in the past/present which should remain corrected after measuring and correcting u
    P_Au = lambda A, u: C(A.union(Lambda_Y(u)).union(Lambda_Z(u)))

    """
    from itertools import groupby

    # Define the equivalence relation
    def is_even(x):
    return x % 2 == 0

    # Create a set of integers
    S = {1, 2, 3, 4, 5, 6, 7, 8, 9}

    # Calculate the set quotient of S with respect to the equivalence relation on evenness
quotient = []
    for _, group in groupby(S, is_even):
    quotient.append(set(group))

    print(quotient)
    ```

    <shell-maker-end-of-prompt>
    The expression `{(x.union(A)) for x in Delta_Y(u)}` generates a set of subsets of the union of `A` and `Y`, where each subset is obtained by taking the union of the set `A` and a subset of `Y` that is adjacent to `u`.

    Assuming that `Y` is a set of vertices in a graph, and `u` is a vertex in `Y`, the expression is not technically the set quotient of `Y` with respect to an equivalence relation. The set quotient of `Y` with respect to an equivalence relation is typically a set of disjoint sets, where each set contains all the vertices in `Y` that are equivalent with respect to the given relation.

    However, if we think of `Y` as a collection of equivalence classes with respect to some equivalence relation, and `Delta_Y(u)` as the equivalence class of `u`, then the expression `{(x.union(A)) for x in Delta_Y(u)}` can be interpreted as the union of the equivalence class of `u` with the equivalence class of `A`. In this sense, the expression can be thought of as a set quotient of sorts.

    So, while the expression may not be a standard set quotient in the strictest sense, it can be used to compute a related set of subsets of the union of `A` and `Y`. It may be useful in certain contexts, such as when working with graphs or other mathematical structures where equivalence classes are not explicitly defined.
    """

    Y_Au = lambda A, u: {(x.union(A)) for x in Lambda_Y(u)}

    # TODO: the product here might not be right - are we dealiing with matricies or graphs or sets or what?
    M_Au_top = graph.subgraph(product(K_Au(A, u), P_Au(A, u))).adjacency_matrix()
    M_Au_bottom = graph.subgraph(product(K_Au(A, u), Y_Au(A, u))).adjacency_matrix()
    M_Au = np.block([M_Au_top, M_Au_bottom])

    N_Gamma = lambda u: None

    for u in set(graph.nodes()) - set(B):
        # TODO: derivce submatrix from gamma?
        subbmatrix1, submatrix2, submatrix3 = None, None, None
        k_XY, k_XZ, k_YZ = None, None, None
        if plane[u] in ["XY", "X", "Y"]:
            # S\hat{\lambda} = [ {u} //  0 ]
            submatrix1 = 0  # TODO
            k_XY = 0  # TODO
        if plane[u] in ["XZ", "X", "Z"]:
            # S\hat{\lambda} = [ N_gamma(u) U P_Au U {u} // N_gamma(u) /\ Y_Au ]
            submatrix2 = 0  # TODO
            k_XZ = 0  # TODO
        if plane[u] in ["YZ", "Y", "Z"]:
            # S\hat{\lambda} = [ N_gamma(u) /\ P_Au // N_gamma(u) /\ Y_Au ]            submatrix3 = 0  # TODO
            submatrix3 = 0
            k_YZ = 0  # TODO

        # TODO is None the right way to handle this?
        if (k_XY is not None) or (k_XZ is not None) or (k_YZ is not None):
            C.add(u)
            sol = k_XY or k_XZ or k_YZ
            p[u] = sol
            d[u] = k

    # If no witness set is found, and all nodes have been measured, then the graph state has Pauli flow.
    if len(C) == 0 and k > 0:
        if set(B) == set(graph.nodes()):
            return True, p, d
        else:
            return False, dict(), dict()
    else:
        B = B.union(C)
        return pflowaux(graph, gamma, inputs, plane, B, B, k + 1, d, p)

# This section implements the dispatch function to emulate method overloading for the find_flow function.
# This is necessary because the GraphState class is contained in an MBQCircuit class,
# so we need to be able to call find_flow on both of them. This is done by checking the type of the resource
# and calling the appropriate function.

def _find_flow_dispatch(
    flow_handler,
    resource: Resource,
    input_nodes=None,
    output_nodes=None,
    basis=None,
) -> FlowResult:
    # TODO circular dependencies of MBQCircuit
    # if isinstance(resource, MBQCircuit):
    # return flow_handler(
    #     resource.graph,
    #     resource.input_nodes,
    #     resource.output_nodes,
    #     resource.default_measurement,
    # )
    #    pass
    if isinstance(resource, GraphState):
        if input_nodes is None or output_nodes is None:
            raise ValueError("Input and output nodes must be specified for GraphState")
    else:
        raise TypeError(
            f"mbqc must be of type MBQCircuit or GraphState, not {type(resource)}"
        )

    return flow_handler(resource, input_nodes, output_nodes)

# TODO: find_flow does not use FlowResult
def find_flow(resource: Resource, input_nodes=None, output_nodes=None) -> object:
    return _find_flow_dispatch(_find_flow, resource, input_nodes, output_nodes)

def find_cflow(resource: Resource, input_nodes=None, output_nodes=None) -> FlowResult:
    return _find_flow_dispatch(_find_cflow, resource, input_nodes, output_nodes)

def find_gflow(resource: Resource, input_nodes=None, output_nodes=None) -> FlowResult:
    return _find_flow_dispatch(_find_gflow, resource, input_nodes, output_nodes)

def find_pflow(
    resource: Resource,
    input_nodes=None,
    output_nodes=None,
    basis: Union[Ment, str, dict] = Ment("XY"),
) -> FlowResult:
    if basis is None:
        raise ValueError("Basis cannot be None")
    return _find_flow_dispatch(
        partial(_find_pflow, basis=basis),
        resource,
        input_nodes,
        output_nodes
    )

def check_if_flow(
    graph: GraphState, input_nodes: List, output_nodes: List, flow, partial_order
) -> bool:
    """Checks if flow satisfies conditions on state."""
    conds = True
    for i in [v for v in graph.nodes() if v not in output_nodes]:
        nfi = list(graph.neighbors(flow(i)))
        c1 = i in nfi
        c2 = partial_order(i, flow(i))
        c3 = math.prod([partial_order(i, k) for k in set(nfi) - {i}])
        conds = conds * c1 * c2 * c3
        # TODO: name the conditions
        if not c1:
            print(f"Condition 1 failed for node {i}. {i} not in {nfi}")
        if not c2:
            print(f"Condition 2 failed for node {i}. {i} ≮ {flow(i)}")
        if not c3:
            print(f"Condition 3 failed for node {i}.")
            for k in set(nfi) - {i}:
                if not partial_order(i, k):
                    print(f"{i} ≮ {k}")
    return conds
