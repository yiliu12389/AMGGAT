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


def parse_spice(path):
    """Parse a resistor/current/voltage netlist into ``A x = b``."""
    node_idx, resistor_edges, voltage_pins, current_loads = {}, [], {}, {}

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
                if j == -1:
                    voltage_pins[i] = value
                elif i == -1:
                    voltage_pins[j] = -value
            elif kind == 'I':
                i, j, value = node_id(parts[1]), node_id(parts[2]), parse_spice_number(parts[3])
                if i != -1:
                    current_loads[i] = current_loads.get(i, 0.0) - value
                if j != -1:
                    current_loads[j] = current_loads.get(j, 0.0) + value

    n_nodes = len(node_idx)
    names = [None] * n_nodes
    for name, index in node_idx.items():
        names[index] = name

    rows, cols, values = [], [], []
    for i, j, conductance in resistor_edges:
        if i == -1 and j == -1:
            continue
        if i == -1:
            rows.append(j); cols.append(j); values.append(conductance)
        elif j == -1:
            rows.append(i); cols.append(i); values.append(conductance)
        else:
            rows.extend((i, j, i, j))
            cols.extend((i, j, j, i))
            values.extend((conductance, conductance, -conductance, -conductance))

    b_vdd = np.zeros(n_nodes, dtype=np.float64)
    b_isource = np.zeros(n_nodes, dtype=np.float64)
    for index, current in current_loads.items():
        b_isource[index] += current
    matrix = sp.coo_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    matrix.sum_duplicates()
    pin_ids = np.asarray([i for i, voltage in voltage_pins.items()
                          if voltage > 0.0], dtype=np.int64)
    pin_values = np.asarray([voltage_pins[i] for i in pin_ids], dtype=np.float64)
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


def parse_layer(names):
    """Extract the physical metal-layer index encoded in each node name."""
    layers = np.zeros(len(names), dtype=np.int64)
    for index, name in enumerate(names):
        if name is not None:
            match = _NAME_RE.match(name)
            if match:
                layers[index] = int(match.group(2))
    return layers


def load_sample(spice_path, solution_group, sample_name=None):
    matrix, rhs, b_vdd, b_isource, names, vdd = parse_spice(spice_path)
    name = sample_name or os.path.splitext(os.path.basename(spice_path))[0]
    if name == 'netlist':
        name = os.path.basename(os.path.dirname(spice_path))
    solution_path = os.path.join(SOL, solution_group, f'{name}.solution')
    if not os.path.exists(solution_path):
        return None
    x_gt, mask = load_gt(solution_path, names)
    if not mask.any():
        return None
    layer_idx = parse_layer(names)
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


def load_group_flat(subdir):
    """Load flat benchmark groups whose netlists end in ``.sp``."""
    base = os.path.join(BENCH, subdir)
    samples = []
    for spice_path in sorted(glob.glob(os.path.join(base, '*.sp'))):
        sample = load_sample(spice_path, subdir)
        if sample is not None:
            samples.append(sample)
    return samples
