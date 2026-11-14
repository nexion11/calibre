# Models that look at whole frames instead of only tool boxes.
#  1) pretrained-embedding model: summarize each trial's frame embeddings, shrink with PCA, then ridge
#  2) SOCALNet-style: per-second ResNet-50 features + tool presence -> LSTM, our re-build of their design
#     (ResNet + LSTM on the first 60 s with tool presence as extra input), trained under OUR surgeon-held-out rules
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from calibre.components.vid import human_boxes_file, load_tool_boxes
from calibre.models.tool_detector_training import tool_types

seconds_kept = 60


def load_embeddings(name):
    saved = np.load(f"data/embeddings_{name}.npz", allow_pickle=True)
    return saved["embeddings"], saved["trial_id"], saved["second"]


def trial_embedding_summary(name):
    """Two numbers per embedding dimension per trial: average over the minute, and average over the last 15 s."""
    embeddings, trial_ids, seconds = load_embeddings(name)
    rows = {}
    for trial_id in np.unique(trial_ids):
        mine = trial_ids == trial_id
        late = mine & (seconds > seconds_kept - 15)
        late_mean = embeddings[late].mean(0) if late.any() else embeddings[mine].mean(0)
        rows[trial_id] = np.concatenate([embeddings[mine].mean(0), late_mean])
    return rows


def embedding_regressor(summaries, extra_features=(), components=10):
    def fit_predict(train, test):
        train_raw = np.stack([summaries[t] for t in train["trial_id"]])
        test_raw = np.stack([summaries[t] for t in test["trial_id"]])
        pca = make_pipeline(StandardScaler(), PCA(n_components=components, random_state=0)).fit(train_raw)
        train_x = np.hstack([pca.transform(train_raw), train[list(extra_features)].values])
        test_x = np.hstack([pca.transform(test_raw), test[list(extra_features)].values])
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
        model.fit(train_x, np.log(train["blood_loss"].values))
        return np.exp(model.predict(test_x))
    return fit_predict


def embedding_failure(summaries, extra_features=(), components=10):
    def fit_predict(train, test):
        train_raw = np.stack([summaries[t] for t in train["trial_id"]])
        test_raw = np.stack([summaries[t] for t in test["trial_id"]])
        pca = make_pipeline(StandardScaler(), PCA(n_components=components, random_state=0)).fit(train_raw)
        train_x = np.hstack([pca.transform(train_raw), train[list(extra_features)].values])
        test_x = np.hstack([pca.transform(test_raw), test[list(extra_features)].values])
        model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, max_iter=2000))
        model.fit(train_x, train["success"].values == 0)
        return model.predict_proba(test_x)[:, 1]
    return fit_predict


# ---------------- SOCALNet-style sequence model ----------------

def sequence_inputs(boxes_file=human_boxes_file):
    """trial_id -> (60 x [2048 ResNet + 5 tool-present flags]) array."""
    embeddings, trial_ids, seconds = load_embeddings("resnet50")
    boxes = load_tool_boxes(boxes_file).dropna(subset=["label"])
    present = set(zip(boxes["trial_id"], boxes["frame_number"], boxes["label"]))
    sequences = {}
    for trial_id in np.unique(trial_ids):
        mine = np.where(trial_ids == trial_id)[0]
        order = mine[np.argsort(seconds[mine])]
        steps = []
        for row in order:
            flags = [float((trial_id, seconds[row], tool) in present) for tool in tool_types]
            steps.append(np.concatenate([embeddings[row], flags]))
        sequences[trial_id] = np.array(steps, dtype=np.float32)
    return sequences


class SequenceNet(torch.nn.Module):
    def __init__(self, input_size, hidden=64):
        super().__init__()
        self.squeeze = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(input_size, hidden), torch.nn.ReLU())
        self.lstm = torch.nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.blood_head = torch.nn.Linear(2 * hidden, 1)
        self.failure_head = torch.nn.Linear(2 * hidden, 1)

    def forward(self, x, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(self.squeeze(x), lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.blood_head(last).squeeze(1), self.failure_head(last).squeeze(1)


class SocalnetStyle:
    """Trains one network per training set and caches it, so the blood and failure heads share one fit."""

    def __init__(self, sequences, epochs=60, seed=0):
        self.sequences, self.epochs, self.seed, self.cache = sequences, epochs, seed, {}

    def _batch(self, trial_ids, mean, std):
        seqs = [(self.sequences[t] - mean) / std for t in trial_ids]
        lengths = torch.tensor([len(s) for s in seqs])
        padded = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(s) for s in seqs], batch_first=True)
        return padded, lengths

    def _fit(self, train):
        key = tuple(sorted(train["trial_id"]))
        if key in self.cache:
            return self.cache[key]
        torch.manual_seed(self.seed)
        stacked = np.concatenate([self.sequences[t] for t in train["trial_id"]])
        mean, std = stacked.mean(0), stacked.std(0) + 1e-6
        x, lengths = self._batch(train["trial_id"].tolist(), mean, std)
        log_blood = torch.tensor(np.log(train["blood_loss"].values), dtype=torch.float32)
        blood_mean, blood_std = log_blood.mean(), log_blood.std()
        failed = torch.tensor((train["success"].values == 0).astype(np.float32))
        net = SequenceNet(x.shape[2])
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        for _ in range(self.epochs):
            net.train()
            for batch in torch.randperm(len(train)).split(16):
                blood_out, failure_out = net(x[batch], lengths[batch])
                loss = torch.nn.functional.mse_loss(blood_out, (log_blood[batch] - blood_mean) / blood_std) \
                    + torch.nn.functional.binary_cross_entropy_with_logits(failure_out, failed[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        net.eval()
        self.cache[key] = (net, mean, std, blood_mean, blood_std)
        return self.cache[key]

    def predict_ml(self, train, test):
        net, mean, std, blood_mean, blood_std = self._fit(train)
        with torch.no_grad():
            blood_out, _ = net(*self._batch(test["trial_id"].tolist(), mean, std))
        return np.exp((blood_out * blood_std + blood_mean).numpy())

    def predict_failure(self, train, test):
        net, mean, std, *_ = self._fit(train)
        with torch.no_grad():
            _, failure_out = net(*self._batch(test["trial_id"].tolist(), mean, std))
        return torch.sigmoid(failure_out).numpy()
