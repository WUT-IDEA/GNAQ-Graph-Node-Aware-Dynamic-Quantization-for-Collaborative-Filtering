import torch
import torch.nn as nn
import numpy as np
from data_loading.data_loader import (
    DataLoader,
    convert_sp_mat_to_sp_tensor,
    convert_sp_tensor_to_sp_mat,
)
import os
import dgl
from timeit import default_timer as timer
from datetime import timedelta
import scipy.sparse as sp

# from set_parse import config_parser
# from utils.util import setup_seed


class BinaryQuantization(nn.Module):
    def __init__(self, opt: dict, data_loader: DataLoader):
        super(BinaryQuantization, self).__init__()
        self.opt = opt
        self.latent_dim = opt["latent_dim"]
        self.field_dims = opt["field_dims"]
        # self.num_clusters = opt["num_clusters"]
        self.num_layers = opt["num_layers"]
        # A: shape N x N
        # self.norm_adj_graph = data_loader.norm_adj_graph
        self.norm_adj_graph = data_loader.norm_adj_graph
        self.norm_adj_graph_tensor = (
            convert_sp_mat_to_sp_tensor(data_loader.norm_adj_graph.tocoo())
            .coalesce()
            .to(opt["device_id"])
        )
        self.gcn_embs = None

        if opt["use_pretrain_init"]:
            print("use pretrain embedding")
            state_dict = torch.load(
                opt["pretrain_state_dict"], map_location=torch.device("cpu")
            )
            state_dict = torch.cat(
                (state_dict["RP_embed"][0], state_dict["RP_embed"][1]), dim=0
            ).to(opt["device_id"])

            # self.centroid_embs = torch.zeros(
            #     (data_loader.m_item() + data_loader.n_user(), self.num_clusters)
            # )
            max_values, _ = torch.max(state_dict, dim=1, keepdim=True)
            min_values, _ = torch.min(state_dict, dim=1, keepdim=True)
            gap = (max_values - min_values) / 5
            self.centroid_embs = torch.cat(
                (
                    min_values + gap,
                    min_values + 2 * gap,
                    min_values + 3 * gap,
                    min_values + 4 * gap,
                ),
                dim=1,
            )

            # assignment_mat = torch.zeros(sum(self.field_dims), self.latent_dim)
            assignment_gap = (max_values - min_values) / 4
            self.static_gap1 = min_values + assignment_gap
            self.static_gap2 = min_values + 2 * assignment_gap
            self.static_gap3 = min_values + 3 * assignment_gap
            mask0 = state_dict <= min_values + assignment_gap
            mask1 = (min_values + assignment_gap < state_dict) & (state_dict <= min_values + 2 * assignment_gap)
            mask2 = (min_values + 2 * assignment_gap < state_dict) & (state_dict <= min_values + 3 * assignment_gap)
            mask3 = state_dict > min_values + 3 * assignment_gap
            state_dict[mask0] = 0
            state_dict[mask1] = 1
            state_dict[mask2] = 2
            state_dict[mask3] = 3
            self.centroid_assignment = state_dict.detach()

            # self.graph = dgl.from_scipy(
            #     self.norm_adj_graph, eweight_name="norm_weight"
            # ).to(opt["device_id"])
            # self.graph.ndata["h"] = state_dict

            # self.graph.update_all(
            #     dgl.function.copy_u("h", "msg"),  # 发送消息：把源节点特征作为消息
            #     self.mean_aggregator,  # 聚合函数：计算消息的均值
            # )
            self.centroid_embs = nn.Parameter(self.centroid_embs)

        self.assignment_save_path = os.path.join(self.opt["res_prepath"], "assignment")
        self.centroid_save_path = os.path.join(self.opt["res_prepath"], "centroids")

        os.makedirs(self.assignment_save_path, exist_ok=True)
        os.makedirs(self.centroid_save_path, exist_ok=True)
        assigment_file_name = os.path.join(self.assignment_save_path, f"init.npz")
        np.savez(assigment_file_name, array=self.centroid_assignment.cpu().numpy())
        centroid_file_name = os.path.join(self.centroid_save_path, "init.npz")
        np.savez(centroid_file_name, array=self.centroid_embs.detach().cpu().numpy())


        print("Init assignment")

    def mean_aggregator(self, nodes):
        return {"h": torch.mean(nodes.mailbox["msg"], dim=1)}

    def forward(self, mode: str = "train"):
        """
        Gather new full embedding, then pass in GCN to get GCN_emb.
        Update emb assignment using matrix approximation thereafter.
        """
        assert mode in ["train", "test"]
        full_embs = self.get_full_embs()
        full_embs = torch.tanh(full_embs)
        self.norm_adj_graph_tensor = self.norm_adj_graph_tensor.to(
            self.opt["device_id"]
        )
        # concat emb in form [full emb, centroid emb]
        # concat_embs = torch.cat((full_embs, self.centroid_embs))

        gcn_embs = [full_embs]
        for _layer in range(self.num_layers):
            full_embs = self.norm_adj_graph_tensor @ full_embs
            gcn_embs.append(full_embs)
        gcn_embs = torch.stack(gcn_embs, dim=1).mean(dim=1)
        self.gcn_embs = gcn_embs
        assert not torch.isnan(gcn_embs).any()

        if mode == "test":
            return gcn_embs
        return full_embs, gcn_embs

    def get_full_embs(self):
        """
        Compute the full embedding table
        """
        N, D = self.centroid_embs.size()
        _, M = self.centroid_assignment.size()
        row_index = torch.arange(N).view(-1, 1).expand(N, M)
        full_embs = self.centroid_embs[row_index, self.centroid_assignment.to(int)]
        return full_embs

    def update_assignment(self):
        assignment = self.compute_assignment(self.gcn_embs)

        self.centroid_assignment = assignment.detach()

        # text = f"Epoch {self.opt['epoch_idx']}, Batch {self.opt['batch_idx']} - assignment update\n"
        # print(text, file=self.opt["performance_fp"], flush=True)
        # print(text)

    def save_embs(self):
        # save updated assignment
        assigment_file_name = os.path.join(
            self.assignment_save_path,
            f"epoch_{self.opt['epoch_idx']:d}_batch_{self.opt['batch_idx']:d}.npz",
        )
        np.savez(assigment_file_name, array=self.centroid_assignment.detach().cpu().numpy())

        # save centroid embs
        centroid_file_name = os.path.join(
            self.centroid_save_path,
            f"epoch_{self.opt['epoch_idx']:d}_batch_{self.opt['batch_idx']:d}.npz",
        )
        np.savez(centroid_file_name, array=self.centroid_embs.detach().cpu().numpy())

    def compute_assignment(self, h_full_embs):
        max_values, _ = torch.max(h_full_embs, dim=1, keepdim=True)
        min_values, _ = torch.min(h_full_embs, dim=1, keepdim=True)

        assignment_gap = (max_values - min_values) / 4
        mask0 = h_full_embs <= min_values + assignment_gap
        mask1 = (min_values + assignment_gap < h_full_embs) & (h_full_embs <= min_values + 2 * assignment_gap)
        mask2 = (min_values + 2 * assignment_gap < h_full_embs) & (h_full_embs <= min_values + 3 * assignment_gap)
        mask3 = h_full_embs > min_values + 3 * assignment_gap
        # mask0 = h_full_embs <= self.static_gap1
        # mask1 = (self.static_gap1 < h_full_embs) & (h_full_embs <= self.static_gap2)
        # mask2 = (self.static_gap2 < h_full_embs) & (h_full_embs <= self.static_gap3)
        # mask3 = h_full_embs > self.static_gap3
        h_full_embs[mask0] = 0
        h_full_embs[mask1] = 1
        h_full_embs[mask2] = 2
        h_full_embs[mask3] = 3
        assignment = h_full_embs

        return assignment


# if __name__ == '__main__':
#     parser = config_parser()
#     opt = vars(parser.parse_args())
#     setup_seed(opt["seed"])
#
#     if torch.cuda.is_available():
#         if not isinstance(opt['device_id'], int) and opt["device_id"].isnumeric():
#             opt["device_id"] = f"cuda:{opt['device_id']}"
#     else:
#         opt["device_id"] = "cpu"
#
#     opt["data_path"] = f"./data/{opt['dataset_name']}"
#     opt["alias"] = f"{opt['dataset_name']}_latent_dim_{opt['latent_dim']:d}_" \
#                    f"num_cluster_{opt['num_clusters']}_seed_{opt['seed']:d}_num_composition_embs_{opt['num_composition_centroid']}" \
#                    f"_lr_{opt['lr']:.0e}_optimizer_weight_decay_{opt['optimizer_weight_decay']}_" \
#                    f"l2_penalty_{opt['l2_penalty_factor']}"
#     opt["res_prepath"] = os.path.join(opt["res_prepath"], opt["alias"])
#
#     data_loader = DataLoader(opt=opt)
#     binary_quantization = BinaryQuantization(opt=opt, data_loader=data_loader)
