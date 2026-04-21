import random
from pathlib import Path
import csv
import shutil

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================

CACHE_DIR = Path("dataset/cache")
DIFF_DIR  = Path("dataset/HRTF_Difference")
SOFA_DIR  = Path("dataset/sofa")

DIFF_SUFFIX = "_HRTFdiff.npy"
SOFA_SUFFIX = "_FreeFieldCompMinPhase_48kHz.sofa"

TARGET_AZ = 270.0
TARGET_EL = 0.0
W_EL = 1.0

# 点云采样数
POINTS = 16384

BATCH_SIZE = 8
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_RATIO = 0.2

NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出目录
OUT_DIR = Path("Basic4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENCE = 10
MIN_DELTA = 1e-4

N_TEST = 20
N_PLOT_SUBJECTS = 3

SEED_LIST = [7,11,21,37,42,58,73,91,123,2024]


# =========================
# Utils
# =========================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wrap_angle_deg(a):

    return (a + 180) % 360 - 180


def find_nearest_direction(az_all, el_all, target_az, target_el, w_el=1.0):

    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el

    idx = int(np.argmin(daz**2 + (w_el * delv)**2))

    return idx


def get_target_index_from_sofa(pid):

    sofa_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"

    with h5py.File(sofa_path, "r") as f:

        src_pos = f["SourcePosition"][:]

    az = src_pos[:,0]
    el = src_pos[:,1]

    return find_nearest_direction(az,el,TARGET_AZ,TARGET_EL)


def collect_ids():

    ids = []

    for x in sorted(CACHE_DIR.glob("P*_X.npy")):

        pid = x.name.replace("_X.npy","")

        if (DIFF_DIR / f"{pid}{DIFF_SUFFIX}").exists():

            ids.append(pid)

    return ids


# =========================
# Dataset
# =========================

class TargetDirHRTFDiffDataset(Dataset):

    def __init__(self, cache_dir, diff_dir, ids):

        self.cache_dir = cache_dir
        self.diff_dir = diff_dir
        self.ids = ids

        self.dir_idx = {pid:get_target_index_from_sofa(pid) for pid in ids}

        pid0 = ids[0]

        d = np.load(self.diff_dir / f"{pid0}{DIFF_SUFFIX}")

        self.F = d.shape[2]
        self.out_dim = self.F*2


    def __len__(self):

        return len(self.ids)


    def __getitem__(self, idx):

        pid = self.ids[idx]

        X = np.load(self.cache_dir / f"{pid}_X.npy").astype(np.float32)

        # ===== 关键修改：采样16384点 =====
        if X.shape[0] > POINTS:

            idx_sample = np.random.choice(X.shape[0], POINTS, replace=False)

            X = X[idx_sample]

        D = np.load(self.diff_dir / f"{pid}{DIFF_SUFFIX}").astype(np.float32)

        i_dir = self.dir_idx[pid]

        D_dir = D[i_dir,:,:]

        y_F2 = np.transpose(D_dir,(1,0)).astype(np.float32)

        y = y_F2.reshape(-1)

        return torch.from_numpy(X), torch.from_numpy(y), pid


# =========================
# Metric
# =========================

def lsd_db(pred, gt):

    if pred.ndim == 1:
        pred = pred.unsqueeze(0)

    if gt.ndim == 1:
        gt = gt.unsqueeze(0)

    B,D = pred.shape

    pred2 = pred.view(B,-1,2)
    gt2   = gt.view(B,-1,2)

    diff = pred2 - gt2

    per_ear = torch.sqrt(torch.mean(diff*diff,dim=1))

    return float(torch.mean(per_ear).cpu().item())


# =========================
# Model
# =========================

class PointNetEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(3,64),
            nn.ReLU(),

            nn.Linear(64,128),
            nn.ReLU(),

            nn.Linear(128,256),
            nn.ReLU()

        )


    def forward(self,x):

        feat = self.mlp(x)

        g,_ = torch.max(feat,dim=1)

        return g


class PointNetRegressor(nn.Module):

    def __init__(self,out_dim):

        super().__init__()

        self.enc = PointNetEncoder()

        self.head = nn.Sequential(

            nn.Linear(256,256),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256,out_dim)

        )


    def forward(self,x):

        g = self.enc(x)

        return self.head(g)


# =========================
# Train
# =========================

def train_one_epoch(model,loader,opt):

    model.train()

    loss_fn = nn.MSELoss()

    mse_sum,lsd_sum,n = 0,0,0

    for X,y,_ in loader:

        X,y = X.to(DEVICE),y.to(DEVICE)

        pred = model(X)

        loss = loss_fn(pred,y)

        opt.zero_grad()

        loss.backward()

        opt.step()

        b = X.size(0)

        mse_sum += loss.item()*b

        lsd_sum += lsd_db(pred,y)*b

        n += b

    return mse_sum/n , lsd_sum/n


@torch.no_grad()

def eval_one_epoch(model,loader):

    model.eval()

    loss_fn = nn.MSELoss()

    mse_sum,lsd_sum,n = 0,0,0

    for X,y,_ in loader:

        X,y = X.to(DEVICE),y.to(DEVICE)

        pred = model(X)

        loss = loss_fn(pred,y)

        b = X.size(0)

        mse_sum += loss.item()*b

        lsd_sum += lsd_db(pred,y)*b

        n += b

    return mse_sum/n , lsd_sum/n


# =========================
# Plot
# =========================

@torch.no_grad()

def plot_random_subjects(model,dataset,n_plot=3):

    model.eval()

    pick = random.sample(range(len(dataset)),k=n_plot)

    for idx in pick:

        X,y_true,pid = dataset[idx]

        X = X.unsqueeze(0).to(DEVICE)

        y_pred = model(X).cpu().numpy().squeeze()

        y_true = y_true.numpy()

        F = dataset.F

        yt = y_true.reshape(F,2)
        yp = y_pred.reshape(F,2)

        freqs = np.arange(F)

        plt.figure()

        plt.plot(freqs,yt[:,0],label="GT L")
        plt.plot(freqs,yp[:,0],label="Pred L")

        plt.plot(freqs,yt[:,1],label="GT R")
        plt.plot(freqs,yp[:,1],label="Pred R")

        plt.title(pid)

        plt.legend()

        plt.show()


# =========================
# Train one seed
# =========================

def train_one_seed(seed,ids_sorted):

    set_seed(seed)

    test_ids = ids_sorted[-N_TEST:]
    trainval_ids = ids_sorted[:-N_TEST]

    trainval_dataset = TargetDirHRTFDiffDataset(CACHE_DIR,DIFF_DIR,trainval_ids)
    test_dataset = TargetDirHRTFDiffDataset(CACHE_DIR,DIFF_DIR,test_ids)

    out_dim = trainval_dataset.out_dim

    n_total = len(trainval_dataset)

    n_val = int(n_total*VAL_RATIO)

    n_train = n_total - n_val

    train_set,val_set = random_split(trainval_dataset,[n_train,n_val])

    train_loader = DataLoader(train_set,BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_set,BATCH_SIZE)
    test_loader = DataLoader(test_dataset,BATCH_SIZE)

    model = PointNetRegressor(out_dim).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(),lr=LR)

    best_val = 1e9

    bad_epochs = 0

    ckpt_path = OUT_DIR / f"seed_{seed}.pt"

    for epoch in range(EPOCHS):

        tr_mse,tr_lsd = train_one_epoch(model,train_loader,opt)

        va_mse,va_lsd = eval_one_epoch(model,val_loader)

        print(seed,epoch,va_lsd)

        if best_val - va_lsd > MIN_DELTA:

            best_val = va_lsd

            bad_epochs = 0

            torch.save(model.state_dict(),ckpt_path)

        else:

            bad_epochs += 1

        if bad_epochs >= PATIENCE:

            break

    model.load_state_dict(torch.load(ckpt_path))

    te_mse,te_lsd = eval_one_epoch(model,test_loader)

    return best_val,te_lsd,ckpt_path


# =========================
# Main
# =========================

def main():

    ids = sorted(collect_ids())

    best_seed = None
    best_val = 1e9
    best_ckpt = None

    for seed in SEED_LIST:

        print("Training seed",seed)

        val_lsd,test_lsd,ckpt = train_one_seed(seed,ids)

        if val_lsd < best_val:

            best_val = val_lsd
            best_seed = seed
            best_ckpt = ckpt

    print("BEST SEED",best_seed)

    shutil.copy(best_ckpt,OUT_DIR/"best_model.pt")

    print("Saved best model to Basic4/best_model.pt")


if __name__ == "__main__":

    main()