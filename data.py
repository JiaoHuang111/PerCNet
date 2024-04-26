import itertools
import random
import sys
import time
from pathlib import Path
from typing import Optional
from torch_cluster import radius_graph
import os
import torch
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms

from jarvis.core.graphs import nearest_neighbor_edges, build_undirected_edgedata
from jarvis.db.figshare import data as jdata
from jarvis.core.specie import chem_data, get_node_attributes

# from torch.utils.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.loader import DataLoader

from tqdm import tqdm
import math
from jarvis.db.jsonutils import dumpjson

from pandarallel import pandarallel
import periodictable
import algorithm

pandarallel.initialize(progress_bar=True)

tqdm.pandas()

torch.set_printoptions(precision=10)


def find_index_array(A, B):
    _, n = B.shape
    index_array = torch.zeros(n, dtype=torch.long)

    for i in range(n):
        idx = torch.where((A == B[:, i].unsqueeze(1)).all(dim=0))[0]
        index_array[i] = idx

    return index_array

class StructureDataset(InMemoryDataset):

    def __init__(self, df, data_path, processdir, target, name, atom_features="atomic_number",
                 id_tag="jid", root='./', transform=None, pre_transform=None, pre_filter=None,
                 mean=None, std=None, normalize=False):

        self.df = df
        self.data_path = data_path
        self.processdir = processdir
        self.target = target
        if target == "bulk modulus":
            self.target = "bulk-modulus"
        if target == "shear modulus":
            self.target = "shear-modulus"
        self.name = name
        self.atom_features = atom_features
        # if target != "bulk modulus" and target != "shear modulus":
        self.id_tag = id_tag
        self.ids = self.df[self.id_tag]
        self.labels = torch.tensor(self.df[self.target]).type(
            torch.get_default_dtype()
        )
        if mean is not None:
            self.mean = mean
        elif normalize:
            self.mean = torch.mean(self.labels)
        else:
            self.mean = 0.0
        if std is not None:
            self.std = std
        elif normalize:
            self.std = torch.std(self.labels)
        else:
            self.std = 1.0

        self.group_id = {
            "H": 0,
            "He": 1,
            "Li": 2,
            "Be": 3,
            "B": 4,
            "C": 0,
            "N": 0,
            "O": 0,
            "F": 5,
            "Ne": 1,
            "Na": 2,
            "Mg": 3,
            "Al": 6,
            "Si": 4,
            "P": 0,
            "S": 0,
            "Cl": 5,
            "Ar": 1,
            "K": 2,
            "Ca": 3,
            "Sc": 7,
            "Ti": 7,
            "V": 7,
            "Cr": 7,
            "Mn": 7,
            "Fe": 7,
            "Co": 7,
            "Ni": 7,
            "Cu": 7,
            "Zn": 7,
            "Ga": 6,
            "Ge": 4,
            "As": 4,
            "Se": 0,
            "Br": 5,
            "Kr": 1,
            "Rb": 2,
            "Sr": 3,
            "Y": 7,
            "Zr": 7,
            "Nb": 7,
            "Mo": 7,
            "Tc": 7,
            "Ru": 7,
            "Rh": 7,
            "Pd": 7,
            "Ag": 7,
            "Cd": 7,
            "In": 6,
            "Sn": 6,
            "Sb": 4,
            "Te": 4,
            "I": 5,
            "Xe": 1,
            "Cs": 2,
            "Ba": 3,
            "La": 8,
            "Ce": 8,
            "Pr": 8,
            "Nd": 8,
            "Pm": 8,
            "Sm": 8,
            "Eu": 8,
            "Gd": 8,
            "Tb": 8,
            "Dy": 8,
            "Ho": 8,
            "Er": 8,
            "Tm": 8,
            "Yb": 8,
            "Lu": 8,
            "Hf": 7,
            "Ta": 7,
            "W": 7,
            "Re": 7,
            "Os": 7,
            "Ir": 7,
            "Pt": 7,
            "Au": 7,
            "Hg": 7,
            "Tl": 6,
            "Pb": 6,
            "Bi": 6,
            "Po": 4,
            "At": 5,
            "Rn": 1,
            "Fr": 2,
            "Ra": 3,
            "Ac": 9,
            "Th": 9,
            "Pa": 9,
            "U": 9,
            "Np": 9,
            "Pu": 9,
            "Am": 9,
            "Cm": 9,
            "Bk": 9,
            "Cf": 9,
            "Es": 9,
            "Fm": 9,
            "Md": 9,
            "No": 9,
            "Lr": 9,
            "Rf": 7,
            "Db": 7,
            "Sg": 7,
            "Bh": 7,
            "Hs": 7
        }

        super(StructureDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.path.join(self.root, self.data_path)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.processdir)

    @property
    def processed_file_names(self):
        return self.name + '.pt'

    def process(self):
        mat_data = torch.load(self.raw_file_names)

        data_list = []
        features = self._get_attribute_lookup(self.atom_features)

        for i in tqdm(range(len(mat_data))):
            z = mat_data[i].x
            mat_data[i].atom_numbers = z

            group_feats = []
            for atom in z:
                group_feats.append(self.group_id[periodictable.elements[int(atom)].symbol])
            group_feats = torch.tensor(np.array(group_feats)).type(torch.LongTensor)
            identity_matrix = torch.eye(10)
            g_feats = identity_matrix[group_feats]
            if len(list(g_feats.size())) == 1:
                g_feats = g_feats.unsqueeze(0)

            f = torch.tensor(features[mat_data[i].atom_numbers.long().squeeze(1)]).type(torch.FloatTensor)
            if len(mat_data[i].atom_numbers) == 1:
                f = f.unsqueeze(0)

            mat_data[i].x = f
            mat_data[i].g_feats = g_feats

            mat_data[i].y = (self.labels[i] - self.mean) / self.std
            mat_data[i].label = self.labels[i]

            data_list.append(mat_data[i])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        mat_data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((mat_data, slices), self.processed_paths[0])

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        max_z = max(v["Z"] for v in chem_data.values())

        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features


def load_radius_graphs(
        df: pd.DataFrame,
        name: str = "dft_3d",
        target: str = "",
        radius: float = 4.0,
        max_neighbors: int = 16,
        cachedir: Optional[Path] = None,
):
    def atoms_to_graph(atoms):
        structure = Atoms.from_dict(atoms)
        sps_features = []
        for ii, s in enumerate(structure.elements):
            feat = list(get_node_attributes(s, atom_features="atomic_number"))
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        edges = nearest_neighbor_edges(atoms=structure, cutoff=radius, max_neighbors=max_neighbors)
        u, v, r = build_undirected_edgedata(atoms=structure, edges=edges)

        data = Data(x=node_features, edge_index=torch.stack([u, v]), edge_attr=r.norm(dim=-1))
        return data

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{target}-radius.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        pass
    else:
        graphs = df["atoms"].parallel_apply(atoms_to_graph).values
        torch.save(graphs, cachefile)


def get_3tuple(ids, images, coords_in_cell, matrix):

    n0_index = list(map(lambda item: item[0], ids))

    n0_image = list(map(lambda item: item[0], images))

    n1_index = list(map(lambda item: item[1], ids))

    n1_image = list(map(lambda item: item[1], images))

    all_pos_ij = []
    all_pos_in0 = []
    all_pos_in1 = []
    all_pos_iref = []
    all_pos_jref_j = []
    tuple_u = []
    tuple_v = []

    for i in range(len(ids)):

        pos_i = coords_in_cell[i]

        n0_index_incell_i = n0_index[i]
        # print(n0_index_incell_i)
        n0_image_i = n0_image[i]
        # print(n0_image_i)
        shifted = np.sum(matrix * n0_image_i[:, np.newaxis], axis=0)
        pos_n0 = shifted + coords_in_cell[n0_index_incell_i]
        # print(pos_n0)
        pos_in0 = pos_n0 - pos_i
        n1_index_incell_i = n1_index[i]
        n1_image_i = n1_image[i]
        # print(n1_image_i)
        shifted = np.sum(matrix * n1_image_i[:, np.newaxis], axis=0)
        pos_n1 = shifted + coords_in_cell[n1_index_incell_i]
        # print(pos_n1)
        pos_in1 = pos_n1 - pos_i
        # print(pos_in1)

        for index_j, j in enumerate(ids[i]):
            tuple_u.append(i)
            tuple_v.append(j)
            shifted = np.sum(matrix * images[i][index_j][:, np.newaxis], axis=0)
            pos_j = shifted + coords_in_cell[j]
            # print('position j:')
            # print(pos_j)
            pos_ij = pos_j - pos_i
            if j == n0_index_incell_i and np.allclose(images[i][index_j], n0_image_i, rtol=1e-5):
                pos_iref = pos_in1
            else:
                pos_iref = pos_in0

            n0_index_incell_j = n0_index[j]

            n0_image_j = n0_image[j] + images[i][index_j]
            # print('n0_image_j')
            # print(n0_image_j)
            shifted = np.sum(matrix * n0_image_j[:, np.newaxis], axis=0)
            pos_n0j = shifted + coords_in_cell[n0_index_incell_j]
            # print('pos_n0j')
            # print(pos_n0j)
            pos_jn0 = pos_n0j - pos_j
            # print('pos_jn0')

            n1_index_incell_j = n1_index[j]
            n1_image_j = n1_image[j] + images[i][index_j]
            shifted = np.sum(matrix * n1_image_j[:, np.newaxis], axis=0)
            pos_n1j = shifted + coords_in_cell[n1_index_incell_j]
            pos_jn1 = pos_n1j - pos_j
            if i == n0_index_incell_j and np.allclose(n0_image_j, [0.0, 0.0, 0.0], rtol=1e-5):
                pos_jref = pos_jn1
            else:
                pos_jref = pos_jn0

            all_pos_ij.append(pos_ij)
            all_pos_in0.append(pos_in0)
            all_pos_in1.append(pos_in1)
            all_pos_iref.append(pos_iref)
            all_pos_jref_j.append(pos_jref)

    # print('all position:')
    all_pos_ij = torch.tensor(all_pos_ij)
    # print('all_pos_ij')
    # print(all_pos_ij.size())  # (N*M_i)*3
    # print(all_pos_ij)
    all_pos_in0 = torch.tensor(all_pos_in0)
    # print('all_pos_in0')
    # print(all_pos_in0.size())  # (N*M_i)*3
    # print(all_pos_in0)
    all_pos_in1 = torch.tensor(all_pos_in1)
    # print('all_pos_in1')
    # print(all_pos_in1.size())  # (N*M_i)*3
    # print(all_pos_in1)
    all_pos_iref = torch.tensor(all_pos_iref)
    # print('all_pos_iref')
    # print(all_pos_iref.size())  # (N*M_i)*3
    # print(all_pos_iref)
    all_pos_jref_j = torch.tensor(all_pos_jref_j)

    a = (all_pos_ij * all_pos_in0).sum(dim=-1)
    b = torch.cross(all_pos_ij, all_pos_in0).norm(dim=-1)
    theta = torch.atan2(b, a)
    theta[theta < 0] = theta[theta < 0] + math.pi
    theta[theta < 1e-5] = 0.0

    dist_in0 = all_pos_in0.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(all_pos_ij, all_pos_in0)
    plane2 = torch.cross(all_pos_in0, all_pos_in1)
    # plane2 = torch.cross(-pos_ji, pos_in1)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * all_pos_in0).sum(dim=-1) / dist_in0
    # b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    phi = torch.atan2(b, a)
    phi[phi < 0] = phi[phi < 0] + math.pi
    phi[phi < 1e-5] = 0.0

    dist_ji = all_pos_ij.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(all_pos_ij, all_pos_jref_j)
    plane2 = torch.cross(all_pos_ij, all_pos_iref)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * all_pos_ij).sum(dim=-1) / dist_ji
    tau = torch.atan2(b, a)
    tau[tau < 0] = tau[tau < 0] + math.pi
    tau[tau < 1e-5] = 0.0
    # print('tau calculated..........')

    tuple_u = torch.tensor(tuple_u).to(torch.int64)
    tuple_v = torch.tensor(tuple_v).to(torch.int64)
    dist = torch.tensor(dist_ji)
    # print('result 3tuple....')
    dist = dist.to(torch.float32)
    theta = theta.to(torch.float32)
    phi = phi.to(torch.float32)
    tau = tau.to(torch.float32)
    return tuple_u, tuple_v, dist, theta, phi, tau


def get_new_3tuple(ids, images, coords_in_cell, matrix):
    n0_index = list(map(lambda item: item[0], ids))
    # print('n0_index:')
    # print(len(n0_index))
    # print(n0_index)
    n0_image = list(map(lambda item: item[0], images))
    # print('n0_image:')
    # print(len(n0_image))
    # print(n0_image)
    n1_index = list(map(lambda item: item[1], ids))
    # print('n1_index:')
    # print(len(n1_index))
    # print(n1_index)
    n1_image = list(map(lambda item: item[1], images))
    all_pos_ij = []
    all_pos_in0 = []
    all_pos_in1 = []
    all_pos_jn0 = []
    all_pos_jn1 = []
    tuple_u = []
    tuple_v = []
    for i in range(len(ids)):

        pos_i = coords_in_cell[i]

        n0_index_incell_i = n0_index[i]
        n0_image_i = n0_image[i]
        # print(n0_image_i)
        shifted = np.sum(matrix * n0_image_i[:, np.newaxis], axis=0)
        pos_n0 = shifted + coords_in_cell[n0_index_incell_i]
        # print(pos_n0)
        pos_in0 = pos_n0 - pos_i
        # print(pos_in0)
        n1_index_incell_i = n1_index[i]
        # print(n1_index_incell_i)
        n1_image_i = n1_image[i]
        # print(n1_image_i)
        shifted = np.sum(matrix * n1_image_i[:, np.newaxis], axis=0)
        pos_n1 = shifted + coords_in_cell[n1_index_incell_i]
        pos_in1 = pos_n1 - pos_i
        for index_j, j in enumerate(ids[i]):
            tuple_u.append(i)
            tuple_v.append(j)
            shifted = np.sum(matrix * images[i][index_j][:, np.newaxis], axis=0)
            pos_j = shifted + coords_in_cell[j]
            pos_ij = pos_j - pos_i
            n0_index_incell_j = n0_index[j]

            n0_image_j = n0_image[j] + images[i][index_j]

            shifted = np.sum(matrix * n0_image_j[:, np.newaxis], axis=0)
            pos_n0j = shifted + coords_in_cell[n0_index_incell_j]
            # print('pos_n0j')
            # print(pos_n0j)
            pos_jn0 = pos_n0j - pos_j
            # print('pos_jn0')
            # print(pos_jn0)
            n1_index_incell_j = n1_index[j]
            n1_image_j = n1_image[j] + images[i][index_j]
            shifted = np.sum(matrix * n1_image_j[:, np.newaxis], axis=0)
            pos_n1j = shifted + coords_in_cell[n1_index_incell_j]
            pos_jn1 = pos_n1j - pos_j
            all_pos_ij.append(pos_ij)
            all_pos_in0.append(pos_in0)
            all_pos_in1.append(pos_in1)
            all_pos_jn0.append(pos_jn0)
            all_pos_jn1.append(pos_jn1)

    # print('all position:')
    all_pos_ij = torch.tensor(all_pos_ij)
    all_pos_in0 = torch.tensor(all_pos_in0)

    all_pos_in1 = torch.tensor(all_pos_in1)
    all_pos_jn0 = torch.tensor(all_pos_jn0)

    all_pos_jn1 = torch.tensor(all_pos_jn1)

    a = (all_pos_ij * all_pos_in0).sum(dim=-1)
    b = torch.cross(all_pos_ij, all_pos_in0).norm(dim=-1)
    theta1 = torch.atan2(b, a)
    theta1[theta1 < 0] = theta1[theta1 < 0] + math.pi
    theta1[theta1 < 1e-5] = 0.0

    a = (all_pos_ij * all_pos_in1).sum(dim=-1)
    b = torch.cross(all_pos_ij, all_pos_in1).norm(dim=-1)
    theta2 = torch.atan2(b, a)
    theta2[theta2 < 0] = theta2[theta2 < 0] + math.pi
    theta2[theta2 < 1e-5] = 0.0
    a = (-all_pos_ij * all_pos_jn0).sum(dim=-1)
    b = torch.cross(-all_pos_ij, all_pos_jn0).norm(dim=-1)
    theta3 = torch.atan2(b, a)
    theta3[theta3 < 0] = theta3[theta3 < 0] + math.pi
    theta3[theta3 < 1e-5] = 0.0
    a = (-all_pos_ij * all_pos_jn1).sum(dim=-1)
    b = torch.cross(-all_pos_ij, all_pos_jn1).norm(dim=-1)
    theta4 = torch.atan2(b, a)
    theta4[theta4 < 0] = theta4[theta4 < 0] + math.pi
    theta4[theta4 < 1e-5] = 0.0
    dist_in0 = all_pos_in0.pow(2).sum(dim=-1).sqrt()
    # dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(all_pos_ij, all_pos_in0)
    plane2 = torch.cross(all_pos_in0, all_pos_in1)
    # plane2 = torch.cross(-pos_ji, pos_in1)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * all_pos_in0).sum(dim=-1) / dist_in0
    # b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    phi = torch.atan2(b, a)
    phi[phi < 0] = phi[phi < 0] + math.pi
    phi[phi < 1e-5] = 0.0

    dist_ji = all_pos_ij.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(all_pos_ij, all_pos_in0)
    plane2 = torch.cross(all_pos_ij, all_pos_jn0)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * all_pos_ij).sum(dim=-1) / dist_ji
    tau1 = torch.atan2(b, a)
    tau1[tau1 < 0] = tau1[tau1 < 0] + math.pi
    tau1[tau1 < 1e-5] = 0.0

    dist_ji = all_pos_ij.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(all_pos_ij, all_pos_in0)
    plane2 = torch.cross(all_pos_ij, all_pos_jn1)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * all_pos_ij).sum(dim=-1) / dist_ji
    tau2 = torch.atan2(b, a)
    tau2[tau2 < 0] = tau2[tau2 < 0] + math.pi
    tau2[tau2 < 1e-5] = 0.0

    dist_ji = all_pos_ij.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(all_pos_ij, all_pos_in1)
    plane2 = torch.cross(all_pos_ij, all_pos_jn0)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * all_pos_ij).sum(dim=-1) / dist_ji
    tau3 = torch.atan2(b, a)
    tau3[tau3 < 0] = tau3[tau3 < 0] + math.pi
    tau3[tau3 < 1e-5] = 0.0
    # print('tau calculated..........')
    # print('calculating tau..........')
    dist_ji = all_pos_ij.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(all_pos_ij, all_pos_in1)
    plane2 = torch.cross(all_pos_ij, all_pos_jn1)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * all_pos_ij).sum(dim=-1) / dist_ji
    tau4 = torch.atan2(b, a)
    tau4[tau4 < 0] = tau4[tau4 < 0] + math.pi
    tau4[tau4 < 1e-5] = 0.0
    # print('tau calculated..........')

    tuple_u = torch.tensor(tuple_u).to(torch.int64)
    tuple_v = torch.tensor(tuple_v).to(torch.int64)
    dist = torch.tensor(dist_ji)
    # print('result 3tuple....')
    tuple_u = torch.cat([tuple_u, tuple_u, tuple_u, tuple_u], dim=0)
    tuple_v = torch.cat([tuple_v, tuple_v, tuple_v, tuple_v], dim=0)
    dist = torch.cat([dist, dist, dist, dist], dim=0)
    dist = dist.to(torch.float32)
    theta = torch.cat([theta1, theta1, theta2, theta2], dim=0)
    phi = torch.cat([theta3, theta4, theta3, theta4], dim=0)
    tau = torch.cat([tau1, tau2, tau3, tau4], dim=0)
    theta = theta.to(torch.float32)
    phi = phi.to(torch.float32)
    tau = tau.to(torch.float32)
    return tuple_u, tuple_v, dist, theta, phi, tau

def load_infinite_graphs(
        df: pd.DataFrame,
        name: str = "dft_3d",
        target: str = "",
        cachedir: Optional[Path] = Path('cache'),
        infinite_funcs=[],
        infinite_params=[],
        R=5,
):
    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)

        # build up atom attribute tensor
        sps_features = []
        for ii, s in enumerate(structure.elements):
            feat = list(get_node_attributes(s, atom_features="atomic_number"))
            sps_features.append(feat)

        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )

        u = torch.arange(0, node_features.size(0), 1).unsqueeze(1).repeat((1, node_features.size(0))).flatten().long()
        v = torch.arange(0, node_features.size(0), 1).unsqueeze(0).repeat((node_features.size(0), 1)).flatten().long()

        edge_index = torch.stack([u, v])

        lattice_mat = structure.lattice_mat.astype(dtype=np.double)

        vecs = structure.cart_coords[u.flatten().numpy().astype(np.int)] - structure.cart_coords[
            v.flatten().numpy().astype(np.int)]

        inf_edge_attr = torch.FloatTensor(np.stack([getattr(algorithm, func)(vecs, lattice_mat, param=param, R=R)
                                     for func, param in zip(infinite_funcs, infinite_params)], 1))
        # print('starting getting neighbor edges')
        edges, all_index, all_images = nearest_neighbor_edges(atoms=structure, cutoff=4, max_neighbors=16)
        # print('starting building undirected edge data')
        u, v, r = build_undirected_edgedata(atoms=structure, edges=edges)
        flag_unique = False
        if flag_unique:
            combined = torch.cat((u.unsqueeze(1), v.unsqueeze(1), r), dim=1)

            unique_combined, unique_indices = torch.unique(combined, dim=0, return_inverse=True)

            unique_u = unique_combined[:, 0]
            unique_v = unique_combined[:, 1]
            unique_r = unique_combined[:, 2:]
            u = unique_u.to(torch.int64)
            v = unique_v.to(torch.int64)
            r = unique_r.to(torch.float32)
        else:
            u = u.to(torch.int64)
            v = v.to(torch.int64)
            r = r.to(torch.float32)

        coords_in_cell = structure.cart_coords
        matrix = np.array(structure.lattice_mat)
        tuple_u, tuple_v, dist, theta, phi, tau = get_new_3tuple(all_index, all_images, coords_in_cell, matrix)
        # tuple_edge_index, dist, theta, phi, tau = get_4_tuple(structure.cart_coords, new_cutoff=4.0)
        # print(test done)
        data = Data(x=node_features, edge_attr=r.norm(dim=-1), edge_index=torch.stack([u, v]), inf_edge_index=edge_index,
                    inf_edge_attr=inf_edge_attr
                    , tuple_edge_index=torch.stack([tuple_u, tuple_v]), dist=dist, theta=theta, phi=phi, tau=tau
                    )
        return data

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{target}-infinite.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        pass
    else:
        #graphs = df["atoms"].progress_apply(atoms_to_graph).values
        graphs = df["atoms"].parallel_apply(atoms_to_graph).values
        torch.save(graphs, cachefile)


def get_id_train_val_test(
        total_size=1000,
        split_seed=123,
        train_ratio=None,
        val_ratio=0.1,
        test_ratio=0.1,
        n_train=None,
        n_test=None,
        n_val=None,
        keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
            train_ratio is None
            and val_ratio is not None
            and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1

    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)

    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test): -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


def get_torch_dataset(
        dataset=None,
        root="",
        cachedir="",
        processdir="",
        name="",
        id_tag="jid",
        target="",
        atom_features="",
        normalize=False,
        euclidean=False,
        cutoff=4.0,
        max_neighbors=16,
        infinite_funcs=[],
        infinite_params=[],
        R=5,
        mean=0.0,
        std=1.0,
):
    """Get Torch Dataset."""
    df = pd.DataFrame(dataset)
    if target == "shear modulus":
        vals = df["shear-modulus"]
    elif target == "bulk modulus":
        vals = df["bulk-modulus"]
    else:
        vals = df[target].values
    print("data range", np.max(vals), np.min(vals))
    cache = os.path.join(root, cachedir)
    if not os.path.exists(cache):
        os.makedirs(cache)
    if euclidean:
        load_radius_graphs(
            df,
            radius=cutoff,
            max_neighbors=max_neighbors,
            name=name + "-" + str(cutoff),
            target=target,
            cachedir=Path(cache),
        )

        data = StructureDataset(
            df,
            os.path.join(cachedir, f"{name}-{cutoff}-{target}-radius.bin"),
            processdir,
            target=target,
            name=f"{name}-{cutoff}-{target}-radius",
            atom_features=atom_features,
            id_tag=id_tag,
            root=root,
            mean=mean,
            std=std,
            normalize=normalize,
        )
    else:
        load_infinite_graphs(
            df,
            name=name,
            target=target,
            cachedir=Path(cache),
            infinite_funcs=infinite_funcs,
            infinite_params=infinite_params,
            R=R,
        )

        data = StructureDataset(
            df,
            os.path.join(cachedir, f"{name}-{target}-infinite.bin"),
            processdir,
            target=target,
            name=f"{name}-{target}-infinite",
            atom_features=atom_features,
            id_tag=id_tag,
            root=root,
            mean=mean,
            std=std,
            normalize=normalize,
        )
    return data


def get_train_val_loaders(
        dataset: str = "dft_3d",
        root: str = "",
        cachedir: str = "",
        processdir: str = "",
        dataset_array=None,
        target: str = "formation_energy_peratom",
        atom_features: str = "cgcnn",
        n_train=None,
        n_val=None,
        n_test=None,
        train_ratio=None,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size: int = 64,
        split_seed: int = 123,
        keep_data_order=False,
        workers: int = 4,
        pin_memory: bool = True,
        id_tag: str = "jid",
        normalize=False,
        euclidean=False,
        cutoff: float = 4.0,
        max_neighbors: int = 16,
        infinite_funcs=[],
        infinite_params=[],
        R=5,
):
    if target == "shear modulus":
        import pickle
        with open('DATA/shear-modulus_train.pkl', 'rb') as f:
            dataset_train = pickle.load(f)
        with open('DATA/shear-modulus_val.pkl', 'rb') as f:
            dataset_val = pickle.load(f)
        with open('DATA/shear-modulus_test.pkl', 'rb') as f:
            dataset_test = pickle.load(f)
    elif target == "bulk modulus":
        import pickle
        with open('DATA/bulk-modulus_train.pkl', 'rb') as f:
            dataset_train = pickle.load(f)
        with open('DATA/bulk-modulus_val.pkl', 'rb') as f:
            dataset_val = pickle.load(f)
        with open('DATA/bulk-modulus_test.pkl', 'rb') as f:
            dataset_test = pickle.load(f)
    else:
        if not dataset_array:
            d = jdata(dataset)
        else:
            d = dataset_array

        dat = []
        all_targets = []

        for i in d:
            if isinstance(i[target], list):
                all_targets.append(torch.tensor(i[target]))
                dat.append(i)

            elif (
                    i[target] is not None
                    and i[target] != "na"
                    and not math.isnan(i[target])
            ):
                dat.append(i)
                all_targets.append(i[target])

        id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(dat),
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            n_train=n_train,
            n_test=n_test,
            n_val=n_val,
            keep_data_order=keep_data_order,
        )
        ids_train_val_test = {}
        ids_train_val_test["id_train"] = [dat[i][id_tag] for i in id_train]
        ids_train_val_test["id_val"] = [dat[i][id_tag] for i in id_val]
        ids_train_val_test["id_test"] = [dat[i][id_tag] for i in id_test]
        dumpjson(
            data=ids_train_val_test,
            filename=os.path.join(root, "ids_train_val_test.json"),
        )
        dataset_train = [dat[x] for x in id_train]
        dataset_val = [dat[x] for x in id_val]
        dataset_test = [dat[x] for x in id_test]

    start = time.time()
    train_data = get_torch_dataset(
        dataset=dataset_train,
        root=root,
        cachedir=cachedir,
        processdir=processdir,
        name=dataset + "_train",
        id_tag=id_tag,
        target=target,
        atom_features=atom_features,
        normalize=normalize,
        euclidean=euclidean,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        infinite_funcs=infinite_funcs,
        infinite_params=infinite_params,
        R=R,
    )

    mean = train_data.mean
    std = train_data.std

    val_data = get_torch_dataset(
        dataset=dataset_val,
        root=root,
        cachedir=cachedir,
        processdir=processdir,
        name=dataset + "_val",
        id_tag=id_tag,
        target=target,
        atom_features=atom_features,
        normalize=normalize,
        euclidean=euclidean,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        infinite_funcs=infinite_funcs,
        infinite_params=infinite_params,
        R=R,
        mean=mean,
        std=std
    )

    test_data = get_torch_dataset(
        dataset=dataset_test,
        root=root,
        cachedir=cachedir,
        processdir=processdir,
        name=dataset + "_test",
        id_tag=id_tag,
        target=target,
        atom_features=atom_features,
        normalize=normalize,
        euclidean=euclidean,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        infinite_funcs=infinite_funcs,
        infinite_params=infinite_params,
        R=R,
        mean=mean,
        std=std,
    )

    print("------processing time------: " + str(time.time() - start))

    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))
    return (
        train_loader,
        val_loader,
        test_loader,
        mean,
        std,
    )
