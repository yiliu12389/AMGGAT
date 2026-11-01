"""SPICE parsing, ground-truth loading, and circuit dataset discovery."""

import glob
import os
import re

import numpy as np
import scipy.sparse as sp

from data.paths import BENCH, SOL

_NAME_RE = re.compile(r'n(\d+)_m(\d+)_(\d+)_(\d+)')
_SPICE_NUMBER = re.compile(
    r'^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]*)$')
_SPICE_SCALE = {'t': 1e12, 'g': 1e9, 'k': 1e3, 'm': 1e-3,
                'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15}


def parse_spice_number(token):
    """Parse a SPICE numeric literal, including common engineering suffixes."""
    match = _SPICE_NUMBER.match(token.strip())
    if match is None:
        raise ValueError(f'Invalid SPICE numeric value: {token!r}')
    value = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix.startswith('meg'):
        return value * 1e6
    if suffix and suffix[0] in _SPICE_SCALE:
        return value * _SPICE_SCALE[suffix[0]]
    return value


def _ibm_vdd_nets(path):
    return {
        net for net, (_, rail) in _ibm_net_metadata(path).items()
        if rail.startswith('VDD')
    }


def _ibm_net_metadata(path):
    """Return ``net index -> (physical metal layer, rail name)`` for IBM."""
    metadata = {}
    with open(path) as spice_file:
        for line in spice_file:
            match = re.match(
                r'\*\s*layer:\s*M(\d+),([^\s]+)\s+net:\s*(\d+)',
                line.strip(), flags=re.IGNORECASE)
            if match:
                metadata[int(match.group(3))] = (
                    int(match.group(1)), match.group(2).upper())
    return metadata


def _node_net_index(name):
    normalized = name[3:] if name.startswith('_X_') else name
    match = re.match(r'^n(\d+)_', normalized)
    return int(match.group(1)) if match else None


def parse_spice(path, vdd_only=False):
    """Parse a resistor/current/voltage netlist into ``A x = b``."""
    node_idx, resistor_edges, zero_voltage_edges = {}, [], []
    voltage_pins, current_loads = {}, {}
    vdd_nets = _ibm_vdd_nets(path) if vdd_only else set()

    def node_id(name):
        if name == '0':
            return -1
        if name not in node_idx:
            node_idx[name] = len(node_idx)
        return node_idx[name]

    with open(path) as spice_file:
        for line in spice_file:
            line = line.strip()
            if not line or line[0] in '*.':
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            kind = parts[0][0].upper()
            if kind == 'R':
                i, j, resistance = node_id(parts[1]), node_id(parts[2]), parse_spice_number(parts[3])
                if resistance > 0:
                    resistor_edges.append((i, j, 1.0 / resistance))
            elif kind == 'V':
                i, j, value = node_id(parts[1]), node_id(parts[2]), parse_spice_number(parts[3])
                if value == 0.0 and i != -1 and j != -1:
                    zero_voltage_edges.append((i, j))
                elif j == -1:
                    voltage_pins[i] = value
                elif i == -1:
                    voltage_pins[j] = -value
            elif kind == 'I':
                i, j, value = node_id(parts[1]), node_id(parts[2]), parse_spice_number(parts[3])
                if i != -1:
                    current_loads[i] = current_loads.get(i, 0.0) - value
                if j != -1:
                    current_loads[j] = current_loads.get(j, 0.0) + value

    n_raw = len(node_idx)
    names = [None] * n_raw
    for name, index in node_idx.items():
        names[index] = name

    keep = np.ones(n_raw, dtype=bool)
    if vdd_nets:
        keep = np.asarray([
            _node_net_index(node_name) in vdd_nets
            for node_name in names
        ], dtype=bool)
    active_ids = np.flatnonzero(keep)
    old_to_active = {int(old): int(new)
                     for new, old in enumerate(active_ids)}
    names = [names[index] for index in active_ids]

    def remap_node(index):
        if index == -1:
            return -1
        return old_to_active.get(index, -1)

    active_resistors = [
        (remap_node(i), remap_node(j), conductance)
        for i, j, conductance in resistor_edges
        if (i == -1 or i in old_to_active) and
           (j == -1 or j in old_to_active)
    ]
    active_shorts = [
        (old_to_active[i], old_to_active[j])
        for i, j in zero_voltage_edges
        if i in old_to_active and j in old_to_active
    ]
    active_pins = {
        old_to_active[index]: value
        for index, value in voltage_pins.items()
        if index in old_to_active
    }
    active_currents = {
        old_to_active[index]: current
        for index, current in current_loads.items()
        if index in old_to_active
    }

    n_active = len(names)
    parent = np.arange(n_active, dtype=np.int64)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for left, right in active_shorts:
        union(left, right)

    roots = {}
    component = np.empty(n_active, dtype=np.int64)
    component_names = []
    for index, node_name in enumerate(names):
        root = find(index)
        if root not in roots:
            roots[root] = len(roots)
            component_names.append(node_name)
        component[index] = roots[root]
    names = component_names
    n_nodes = len(names)

    rows, cols, values = [], [], []
    for i, j, conductance in active_resistors:
        i = int(component[i]) if i != -1 else -1
        j = int(component[j]) if j != -1 else -1
        if i == -1 and j == -1:
            continue
        if i == -1:
            rows.append(j); cols.append(j); values.append(conductance)
        elif j == -1:
            rows.append(i); cols.append(i); values.append(conductance)
        else:
            rows.extend((i, j, i, j))
            cols.extend((i, j, j, i))
            values.extend((conductance, conductance,
                           -conductance, -conductance))

    b_vdd = np.zeros(n_nodes, dtype=np.float64)
    b_isource = np.zeros(n_nodes, dtype=np.float64)
    for index, current in active_currents.items():
        b_isource[component[index]] += current
    matrix = sp.coo_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    matrix.sum_duplicates()
    collapsed_pins = {}
    for index, voltage in active_pins.items():
        collapsed_pins[int(component[index])] = voltage
    pin_ids = np.asarray([i for i, voltage in collapsed_pins.items()
                          if voltage > 0.0], dtype=np.int64)
    pin_values = np.asarray([collapsed_pins[i] for i in pin_ids], dtype=np.float64)
    vdd = float(pin_values.max()) if pin_values.size else 1.0
    free_mask = np.ones(n_nodes, dtype=bool)
    free_mask[pin_ids] = False
    free_ids = np.flatnonzero(free_mask)
    b_vdd_free = np.zeros(free_ids.size, dtype=np.float64)
    if pin_ids.size:
        b_vdd_free = -np.asarray(
            matrix[free_ids, :][:, pin_ids] @ pin_values).ravel()
    matrix_free = matrix[free_ids, :][:, free_ids].tocsr()
    b_isource_free = b_isource[free_ids]
    rhs_free = b_vdd_free + b_isource_free
    names_free = [names[index] for index in free_ids]
    return matrix_free, rhs_free, b_vdd_free, b_isource_free, names_free, vdd


def load_gt(gt_path, names):

    values = np.zeros(len(names), dtype=np.float64)
    mask = np.zeros(len(names), dtype=bool)
    if not os.path.exists(gt_path):
        return values, mask
    name_to_index = {name: index for index, name in enumerate(names)}
    with open(gt_path) as solution_file:
        for line in solution_file:
            parts = line.split()
            if len(parts) < 2:
                continue
            index = name_to_index.get(parts[0])
            if index is None:
                continue
            try:
                values[index] = float(parts[1])
                mask[index] = True
            except ValueError:
                continue
    return values, mask


def parse_layer(names, ibm_net_metadata=None):
    """Extract the physical metal-layer index encoded in each node name."""
    layers = np.zeros(len(names), dtype=np.int64)
    for index, name in enumerate(names):
        if name is not None:
            normalized = name
            if normalized.startswith('_X_'):
                normalized = normalized[3:]
            match = _NAME_RE.match(normalized)
            if match:
                layers[index] = int(match.group(2))
                continue
            ibm_match = re.match(r'^n(\d+)_', normalized)
            if ibm_match:
                net_index = int(ibm_match.group(1))
                if ibm_net_metadata is not None:
                    layers[index] = ibm_net_metadata.get(
                        net_index, (0, ''))[0]
                else:
                    layers[index] = net_index
    return layers


def load_sample(spice_path, solution_group, sample_name=None):
    matrix, rhs, b_vdd, b_isource, names, vdd = parse_spice(
        spice_path, vdd_only=(solution_group == 'ibm-circuit-data'))
    name = sample_name or os.path.splitext(os.path.basename(spice_path))[0]
    if name == 'netlist':
        name = os.path.basename(os.path.dirname(spice_path))
    solution_path = os.path.join(SOL, solution_group, f'{name}.solution')
    if not os.path.exists(solution_path):
        return None
    x_gt, mask = load_gt(solution_path, names)
    if not mask.any():
        return None
    ibm_metadata = (_ibm_net_metadata(spice_path)
                    if solution_group == 'ibm-circuit-data' else None)
    layer_idx = parse_layer(names, ibm_net_metadata=ibm_metadata)
    return (name, matrix, rhs, b_vdd, b_isource, layer_idx, x_gt, mask, vdd)


def load_group(subdir):
    """Load circuit subfolders containing ``netlist.sp``."""
    base = os.path.join(BENCH, subdir)
    samples = []
    if not os.path.isdir(base):
        return samples
    for testcase in sorted(os.listdir(base)):
        spice_path = os.path.join(base, testcase, 'netlist.sp')
        if os.path.isfile(spice_path):
            sample = load_sample(spice_path, subdir, sample_name=testcase)
            if sample is not None:
                samples.append(sample)
    return samples


def iter_group_flat(subdir, names=None):
    """Yield flat benchmark samples with either ``.sp`` or ``.spice`` files."""
    base = os.path.join(BENCH, subdir)
    names = None if names is None else set(names)
    spice_paths = sorted(glob.glob(os.path.join(base, '*.sp')))
    spice_paths.extend(sorted(glob.glob(os.path.join(base, '*.spice'))))
    for spice_path in spice_paths:
        sample_name = os.path.splitext(os.path.basename(spice_path))[0]
        if names is not None and sample_name not in names:
            continue
        sample = load_sample(spice_path, subdir)
        if sample is not None:
            yield sample


def load_group_flat(subdir):
    """Load all samples from a flat benchmark group into a list."""
    return list(iter_group_flat(subdir))
