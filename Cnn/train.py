import argparse, os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import EuroSAT
from torchvision import transforms
import wandb

# ---------------- reproducibility ------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- model ---------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

# ---------------- data ----------------------------------------------
def make_loaders(bs, workers=4):
    tf_eval  = transforms.ToTensor()
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    full = EuroSAT(root="data", download=True, transform=tf_eval)
    n_tr, n_val = int(0.7*len(full)), int(0.15*len(full))
    n_te = len(full) - n_tr - n_val
    tr, va, te = random_split(full, [n_tr, n_val, n_te],
                              generator=torch.Generator().manual_seed(SEED))
    tr.dataset.transform = tf_train
    return (DataLoader(tr, bs, shuffle=True,  num_workers=workers),
            DataLoader(va, bs, shuffle=False, num_workers=workers),
            DataLoader(te, bs, shuffle=False, num_workers=workers))

# ---------------- util ------------------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0., 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            preds  = logits.argmax(1)
            loss_sum += loss.item() * x.size(0)
            correct  += (preds==y).sum().item()
            total    += x.size(0)
    return loss_sum/total, correct/total

# ---------------- main ------------------------------------------------
def main():
    ap = argparse.ArgumentParser("EuroSAT trainer (clean W&B)")
    ap.add_argument("--epochs",   type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch",    type=int, default=64)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--quick",    action="store_true")
    ap.add_argument("--project",  default="eurosat")
    ap.add_argument("--entity",   default=None)
    args = ap.parse_args()
    if args.quick: args.epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_dl, va_dl, te_dl = make_loaders(args.batch)
    model = SimpleCNN(10).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    crit  = nn.CrossEntropyLoss()

    run = wandb.init(project=args.project, entity=args.entity,
                     config=vars(args), name="cnn_run", save_code=True)
    wandb.watch(model, log="all", log_freq=100)

    #  epoch is our x-axis for high-level metrics
    wandb.define_metric("epoch")
    for k in ["train/loss_epoch","train/acc_epoch","val/loss","val/acc","grad_norm","lr",
              "test/loss","test/acc"]:
        wandb.define_metric(k, step_metric="epoch")
    # per-batch (optional, can be hidden in UI)
    wandb.define_metric("train/loss_step", step_metric="step")
    wandb.define_metric("train/acc_step",  step_metric="step")

    best, stale, gstep = float("inf"), 0, 0
    for epoch in range(1, args.epochs+1):
        # ---------- train ----------
        model.train()
        run_loss, run_corr, run_tot = 0., 0, 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss   = crit(logits, y)
            loss.backward(); opt.step()

            preds = logits.argmax(1)
            run_loss += loss.item()*x.size(0)
            run_corr += (preds==y).sum().item()
            run_tot  += x.size(0)

            # per-batch logging (commit=False keeps epoch line intact)
            wandb.log({"step": gstep,
                       "train/loss_step": loss.item(),
                       "train/acc_step":  (preds==y).float().mean().item(),
                       "lr": opt.param_groups[0]["lr"]},
                       commit=False)
            gstep += 1

        tr_loss = run_loss/run_tot
        tr_acc  = run_corr/run_tot

        # ---------- validation ----------
        va_loss, va_acc = evaluate(model, va_dl, crit, device)

        # ---------- gradient norm ----------
        with torch.no_grad():
            grad_sq_sum = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_sq_sum += p.grad.detach().pow(2).sum().item()
            grad_norm = grad_sq_sum ** 0.5

        # ---------- epoch-level log ----------
        wandb.log({
            "epoch": epoch,
            "train/loss_epoch": tr_loss,
            "train/acc_epoch":  tr_acc,
            "val/loss": va_loss,
            "val/acc": va_acc,
            "grad_norm": grad_norm
        })

        print(f"Epoch {epoch:02d} | "
              f"train {tr_loss:.4f}/{tr_acc*100:.2f}% | "
              f"val {va_loss:.4f}/{va_acc*100:.2f}%")

        # early-stopping
        if va_loss < best:
            best, stale = va_loss, 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop after {args.patience} stale epochs"); break

    # ---------- test ----------
    te_loss, te_acc = evaluate(model, te_dl, crit, device)
    wandb.log({"epoch": epoch, "test/loss": te_loss, "test/acc": te_acc})
    print(f"TEST | loss {te_loss:.4f} | acc {te_acc*100:.2f}%")
    run.finish()

if __name__ == "__main__":
    main()
