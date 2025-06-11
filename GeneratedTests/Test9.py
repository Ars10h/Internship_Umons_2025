import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, precision_score, recall_score
import logging
import argparse

# ========== CONFIGURATION ==========

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["MLP", "CNN"], default="MLP")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5])
    parser.add_argument("--defenses", nargs="+", type=str, default=["confreg", "dp+confreg", "dp+confreg+secureagg"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

# ========== UTILITAIRES ==========

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== MODÈLES ==========

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(x)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),       # [B,32,13,13]
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),       # [B,64,5,5]
            nn.Flatten()           # [B, 64*5*5 = 1600]
        )
        self.fc = nn.Sequential(
            nn.Linear(1600, 128),  # Taille cohérente
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def get_model(name):
    return MLP() if name == "MLP" else SimpleCNN()

# ========== DÉFENSES ==========

CLIP_NORM = 1.0
CONFIDENCE_REG_WEIGHT = 0.5
SECURE_NOISE = 0.1

def apply_confreg(loss, preds):
    conf = torch.softmax(preds, dim=1).max(dim=1)[0]
    return loss + CONFIDENCE_REG_WEIGHT * conf.mean()

def apply_dp(model, epsilon):
    for p in model.parameters():
        if p.grad is not None:
            p.grad += torch.randn_like(p.grad) * epsilon

def secure_aggregate(models):
    return [
        {k: v + torch.randn_like(v) * SECURE_NOISE for k, v in m.items()}
        for m in models
    ]

# ========== ENTRAÎNEMENT LOCAL ==========

def train_local(model, loader, defense, epsilon, device):
    model.train()
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            if "confreg" in defense:
                loss = apply_confreg(loss, preds)
            loss.backward()
            if "dp" in defense:
                apply_dp(model, epsilon)
            opt.step()
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

# ========== AGRÉGATION ==========

def aggregate(models, defense):
    if "secureagg" in defense:
        models = secure_aggregate(models)
    avg = {k: sum(m[k] for m in models) / len(models) for k in models[0]}
    return avg

# ========== ÉVALUATION ==========

def evaluate(model, loader, device):
    model.eval()
    model.to(device)
    total, correct, all_y, all_pred = 0, 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_y.append(y.cpu())
            all_pred.append(pred.cpu())
    all_y = torch.cat(all_y)
    all_pred = torch.cat(all_pred)
    acc = correct / total
    f1 = f1_score(all_y, all_pred, average="macro")
    prec = precision_score(all_y, all_pred, average="macro")
    rec = recall_score(all_y, all_pred, average="macro")
    return acc, f1, prec, rec

# ========== MIA (MEMBERSHIP INFERENCE ATTACK) ==========

def mia(model, member_x, member_y, nonmember_x, nonmember_y, device):
    crit = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    with torch.no_grad():
        loss_m = crit(model(member_x.to(device)), member_y.to(device))
        loss_nm = crit(model(nonmember_x.to(device)), nonmember_y.to(device))
    threshold = (loss_m.mean() + loss_nm.mean()) / 2
    tpr = (loss_m < threshold).float().mean().item()
    fpr = (loss_nm < threshold).float().mean().item()
    return tpr, fpr, loss_m.cpu().numpy(), loss_nm.cpu().numpy()

def compute_roc_auc(loss_m, loss_nm):
    scores = np.concatenate([-loss_m, -loss_nm])
    labels = np.concatenate([np.ones_like(loss_m), np.zeros_like(loss_nm)])
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr, auc(fpr, tpr)

# ========== FÉDÉRATION ==========

def federated_learning(defense, epsilon, model_name, device, clients, args):
    global_model = get_model(model_name)
    updates = []
    for client_data in clients:
        model = get_model(model_name)
        model.load_state_dict(global_model.state_dict())
        loader = DataLoader(client_data, batch_size=args.batch_size, shuffle=True)
        updated = train_local(model, loader, defense=defense, epsilon=epsilon, device=device)
        updates.append(updated)
    global_model.load_state_dict(aggregate(updates, defense))
    return global_model

# ========== MAIN PIPELINE ==========

def main(args):
    logging.basicConfig(level=logging.INFO)
    seed_all(args.seed)
    device = get_device()
    logging.info(f"Using device: {device}")

    # Préparation des données
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    def split_data(dataset, num_clients):
        indices = np.random.permutation(len(dataset))
        split = np.array_split(indices, num_clients)
        return [Subset(dataset, s) for s in split]

    clients = split_data(train_data, args.num_clients)
    test_loader = DataLoader(test_data, batch_size=128)
    member_batch = next(iter(DataLoader(clients[0], batch_size=64)))
    nonmember_batch = next(iter(test_loader))

    results = []

    for defense in args.defenses:
        for eps in args.epsilons:
            logging.info(f"Défense: {defense} | Epsilon: {eps}")
            model = federated_learning(defense, eps, args.model, device, clients, args)
            acc, f1, prec, rec = evaluate(model, test_loader, device)
            tpr, fpr, loss_m, loss_nm = mia(model, *member_batch, *nonmember_batch, device)
            results.append({
                "defense": defense, "epsilon": eps, "accuracy": acc,
                "f1": f1, "precision": prec, "recall": rec,
                "tpr": tpr, "fpr": fpr, "model": model,
                "loss_m": loss_m, "loss_nm": loss_nm
            })

    # ========== VISUALISATION ==========

    metrics = ["accuracy", "f1", "precision", "recall", "tpr", "fpr"]
    colors = sns.color_palette("tab10", n_colors=len(args.defenses))
    plt.figure(figsize=(14, 12))
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i+1)
        for j, defense in enumerate(args.defenses):
            y = [r[metric] for r in results if r["defense"] == defense]
            x = args.epsilons
            plt.plot(x, y, marker='o', label=defense, color=colors[j])
        plt.title(metric.upper())
        plt.ylabel("Valeur")
        if i >= 4:
            plt.xlabel("Epsilon (bruit)")
        plt.grid(True)
        if i == 0:
            plt.legend(ncol=2, bbox_to_anchor=(0.5, 1.2), loc='upper center')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Défenses optimisées contre MIA", fontsize=16)
    plt.show()

    # Courbes ROC/AUC pour epsilon = 0.3
    plt.figure(figsize=(8, 6))
    for j, defense in enumerate(args.defenses):
        r = next(r for r in results if r['defense'] == defense and np.isclose(r['epsilon'], 0.3))
        fpr, tpr, roc_auc = compute_roc_auc(r['loss_m'], r['loss_nm'])
        plt.plot(fpr, tpr, label=f"{defense} (AUC={roc_auc:.2f})", color=colors[j])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.title("Courbe ROC - MIA (ε = 0.3)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Suggestion automatique finale
    best_combo = max(results, key=lambda x: x['accuracy'] - (x['tpr'] - x['fpr']))
    print("\n>>> Suggestion optimale :")
    print("Défense:", best_combo['defense'], "| Epsilon:", best_combo['epsilon'])
    print("Accuracy: {:.3f}, F1: {:.3f}, TPR: {:.3f}, FPR: {:.3f}".format(
        best_combo['accuracy'], best_combo['f1'], best_combo['tpr'], best_combo['fpr']))

if __name__ == "__main__":
    args = parse_args()
    main(args)
