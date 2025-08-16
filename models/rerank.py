import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loading.data_loader import DataLoader, shuffle, minibatch
from models.recsys import BasicRecSys
import numpy as np

from utils.util import print_text, get_label, recall_atk, ndcg_atk_r


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        if input_dim != hidden_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        batch_size, num_items, dim = x.size()
        x_flat = x.view(-1, dim)  # reshape为(batch_size * num_items, dim)
        residual = self.shortcut(x_flat)
        out = F.relu(self.bn1(self.linear1(x_flat)))
        out = self.bn2(self.linear2(out))
        out += residual
        out = F.relu(out)

        return out.view(batch_size, num_items, -1)


class EmbeddingMLPRanker(nn.Module):
    def __init__(self, embedding_dim, hidden_dims=[512, 768, 512, 256, 128, 64]):
        super(EmbeddingMLPRanker, self).__init__()
        input_dim = embedding_dim * 2

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(input_dim, hidden_dim))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_embedding, item_embedding):
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        logits = self.mlp(x).squeeze(-1)
        return logits


def generate_train_samples(dataloader: DataLoader):
    user_num = dataloader.n_user
    users = np.arange(user_num)
    train_all_pos = dataloader.train_all_pos
    test_all_pos = dataloader.test_all_pos

    # samples = []
    # for i, user in enumerate(users):
    #     user_train_pos = train_all_pos[user]
    #     samples.append([user, user_train_pos, test_all_pos])
    # samples = np.array(samples).astype(np.int32)
    return users, train_all_pos, test_all_pos


# 定义LambdaLoss
def lambda_loss(preds, labels):
    # preds_diff = preds.unsqueeze(2) - preds.unsqueeze(1)
    # labels_diff = labels.unsqueeze(2) - labels.unsqueeze(1)
    #
    # pos_pairs = (labels_diff > 0).float()
    # neg_pairs = (labels_diff < 0).float()
    #
    # S_ij = pos_pairs - neg_pairs
    # lambda_ij = 0.5 * (1 - S_ij) - torch.sigmoid(-preds_diff * S_ij)
    #
    # loss = torch.mean(lambda_ij ** 2)
    loss_fn = nn.MSELoss()
    loss = loss_fn(preds, labels)
    return loss

def train_rerank(
        rerank_model,
        dataloader: DataLoader,
        recsys: BasicRecSys,
        optimizer: torch.optim.Optimizer,
        opt: dict,
):
    device_id = opt["device_id"]
    item_count = opt["field_dims"][1]
    max_k = max(opt["ks"])

    _, train_all_pos, test_all_pos = generate_train_samples(dataloader=dataloader)
    # pos_items = torch.from_numpy(train_all_pos).long().to(device_id)
    # ground_true = torch.from_numpy(test_all_pos).long().to(device_id)
    users = list(range(opt['field_dims'][0]))
    np.random.shuffle(users)

    rerank_model.to(device_id)
    rerank_model.train()

    rating_list = []
    users_list = []
    ground_true_list = []

    batch_count = len(users) // opt["bpr_batch"] + 1
    avg_epoch_loss = 0.
    for idx, batch_user in enumerate(minibatch(users, batch_size=opt["bpr_batch"])):
        opt["first_time_evaluating"] = idx == 0

        # all positive rated items by batch_users, list of lists of item IDs
        all_train_pos = [train_all_pos[_] for _ in batch_user]
        ground_true = [test_all_pos.get(_, []) for _ in batch_user]
        batch_user = torch.tensor(batch_user).long().to(device_id)

        ratings = []
        for u in batch_user:
            u_repeated = torch.full((item_count,), u, device=device_id)
            u_rating = recsys(u_repeated, torch.arange(item_count, device=device_id))
            ratings.append(u_rating)
        # rating dim: batch_user x all_items in the dataset
        ratings = torch.vstack(ratings)

        # exclude positively rated items in training step
        exclude_index = []
        exclude_items = []
        for _, items in enumerate(all_train_pos):
            exclude_index.extend([_] * len(items))
            exclude_items.extend(items)
        ratings[exclude_index, exclude_items] = -(1 << 10)

        # rating_k: max_k of items IDs that has the highest ranking
        _, rating_k = torch.topk(ratings, k=max_k)
        # users_list.append(batch_user)
        # rating_list.append(rating_k.cpu())
        match_mask = torch.zeros_like(rating_k, dtype=torch.float32)
        for i in range(rating_k.size(0)):
            user_ground_true = torch.tensor(ground_true[i], device=rating_k.device)
            match_mask[i] = torch.isin(rating_k[i], user_ground_true).float()

        rerank_k = 10
        batch_user_embeddings = recsys.embedding(batch_user)
        batch_item_embeddings = recsys.embedding(rating_k[:, :rerank_k] + opt['field_dims'][0])
        batch_user_embeddings_expanded = batch_user_embeddings.unsqueeze(1).expand(-1, rerank_k, -1)
        preds = rerank_model(batch_user_embeddings_expanded.detach(), batch_item_embeddings.detach())

        loss = lambda_loss(preds, match_mask[:, :rerank_k])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_text(opt["performance_fp"], f"rerank loss: {loss.item()}")

    rerank_model = rerank_model.cpu()
    torch.save(rerank_model, os.path.join(opt["res_prepath"], "rerank_model.pt"))
    print_text(opt["performance_fp"], "Rerank model saved.")
    rerank_model = rerank_model.to(device_id)


def pool_init_worker(opt: dict):
    global ks
    ks = opt["ks"]


def test_subprocess(X):
    """
    :param X: [(top k rating list, groundTrue_list)] for all users in current process
    """
    sorted_items = X[0].numpy()
    ground_true = X[1]
    # r: Test data ranking & top-k ranking
    r = get_label(ground_true, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in ks:
        rec = recall_atk(ground_true, r, k)
        recall.append(rec)
        ndcg.append(ndcg_atk_r(ground_true, r, k))
    return {'recall': np.array(recall),
            'ndcg': np.array(ndcg)}


def eval_rerank(
        rerank_model,
        dataloader: DataLoader,
        recsys: BasicRecSys,
        opt: dict,
):
    device_id = opt["device_id"]
    train_all_pos = dataloader.train_all_pos
    test_all_pos = dataloader.test_all_pos
    item_count = opt["field_dims"][1]
    test_batch_size = opt["test_batch"]
    recsys.eval()
    max_k = max(opt["ks"])
    # values @ k = 5, 10, 20, 50
    results = {
        "ndcgs": np.zeros(len(opt["ks"])),
        "recalls": np.zeros(len(opt["ks"]))
    }

    with torch.no_grad():
        users = list(range(opt['field_dims'][0]))
        try:
            assert test_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        ground_true_list = []

        batch_count = len(users) // test_batch_size + 1
        for idx, batch_users in enumerate(minibatch(users, batch_size=test_batch_size)):
            # the flag used to retrieve the full embedding for evaluating, will
            # be turned off so long as recsys() is called once
            opt["first_time_evaluating"] = idx == 0

            # all positive rated items by batch_users, list of lists of item IDs
            all_train_pos = [train_all_pos[_] for _ in batch_users]

            # ground true: all items positively rated in Test set
            ground_true = [test_all_pos.get(_, []) for _ in batch_users]
            batch_users = torch.tensor(batch_users).long().to(opt["device_id"])

            ratings = []
            for u in batch_users:
                u_repeated = torch.full((item_count,), u, device=opt["device_id"])
                u_rating = recsys(u_repeated, torch.arange(item_count, device=opt["device_id"]))
                ratings.append(u_rating)
            # rating dim: batch_user x all_items in the dataset
            ratings = torch.vstack(ratings)

            # exclude positively rated items in training step
            exclude_index = []
            exclude_items = []
            for _, items in enumerate(all_train_pos):
                exclude_index.extend([_] * len(items))
                exclude_items.extend(items)
            ratings[exclude_index, exclude_items] = -(1 << 10)

            # rating_k: max_k of items IDs that has the highest ranking
            _, rating_k = torch.topk(ratings, k=max_k)
            users_list.append(batch_users)

            rerank_k = 10
            batch_user_embeddings = recsys.embedding(batch_users)
            batch_item_embeddings = recsys.embedding(rating_k[:, :rerank_k] + opt['field_dims'][0])
            batch_user_embeddings_expanded = batch_user_embeddings.unsqueeze(1).expand(-1, rerank_k, -1)
            preds = rerank_model(batch_user_embeddings_expanded.detach(), batch_item_embeddings.detach())

            sorted_scores, sorted_indices = torch.sort(preds, dim=1, descending=True)
            reranked_top10 = torch.gather(rating_k[:, :rerank_k], 1, sorted_indices)
            # reranked = torch.flip(rating_k[:, :10], dims=[1])
            reranked_rating_k = torch.cat([reranked_top10, rating_k[:, rerank_k:]], dim=1)

            # 之后用reranked_rating_k替换原始rating_k即可

            rating_list.append(reranked_rating_k.cpu())
            ground_true_list.append(ground_true)
        assert batch_count == len(users_list) == len(rating_list) == len(ground_true_list)

        batch_process_input = zip(rating_list, ground_true_list)
        pool_init_worker(opt)
        computed_res = []
        for x in batch_process_input:
            computed_res.append(test_subprocess(x))

        for res in computed_res:
            results["recalls"] += res["recall"]
            results["ndcgs"] += res["ndcg"]
        results['recalls'] /= float(len(test_all_pos.keys()))
        results['ndcgs'] /= float(len(test_all_pos.keys()))
        results["recalls"], results["ndcgs"] = \
            results["recalls"].tolist(), results['ndcgs'].tolist()
    return results


# ==== 使用示例及LambdaLoss训练代码 ====
if __name__ == "__main__":
    embedding_dim = 64
    batch_size = 32
    lr = 0.001

    # 假设预训练好的用户和商品embedding
    user_embeddings = torch.rand(batch_size, embedding_dim)
    item_embeddings = torch.rand(batch_size, embedding_dim)
    labels = torch.randint(0, 2, (batch_size,), dtype=torch.float32)

    # 模型初始化
    model = EmbeddingMLPRanker(embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # 定义LambdaLoss
    def lambda_loss(preds, labels):
        preds_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
        labels_diff = labels.unsqueeze(1) - labels.unsqueeze(0)

        pos_pairs = (labels_diff > 0).float()
        neg_pairs = (labels_diff < 0).float()

        S_ij = pos_pairs - neg_pairs
        lambda_ij = 0.5 * (1 - S_ij) - torch.sigmoid(-preds_diff * S_ij)

        loss = torch.sum(lambda_ij ** 2)
        return loss


    # 训练步骤示例
    model.train()
    optimizer.zero_grad()

    preds = model(user_embeddings, item_embeddings)
    loss = lambda_loss(preds, labels)

    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item()}")
