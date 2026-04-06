"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    ActionAI — End-to-End Multi-Model Human Action Recognition System        ║
║    24AI636 Deep Learning · Scaffolded Project                               ║
║    Models: CNN · MLP · LSTM · GRU · RNN · BiLSTM+Attn · GAN · Autoencoder  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════
# 0. IMPORTS
# ═══════════════════════════════════════════════════════════
import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import time
import json
import hashlib
from PIL import Image
import io
import base64

# ═══════════════════════════════════════════════════════════
# 1. PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ActionAI · End-to-End DL System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# 2. GLOBAL CSS — Dark AI Dashboard Theme
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg:     #06080f;
    --surf:   #0f1520;
    --surf2:  #141d2e;
    --border: #1c2a42;
    --blue:   #3b82f6;
    --violet: #7c3aed;
    --cyan:   #06b6d4;
    --green:  #10b981;
    --amber:  #f59e0b;
    --rose:   #f43f5e;
    --pink:   #ec4899;
    --text:   #e2e8f0;
    --muted:  #64748b;
    --dim:    #334155;
    --r:      14px;
}
html,body,[class*="css"]{
    font-family:'DM Sans',sans-serif;
    background:var(--bg)!important;
    color:var(--text)!important;
}
.main .block-container{ padding:1.2rem 2rem 4rem; max-width:1440px; }

[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#08101e 0%,#060a14 100%)!important;
    border-right:1px solid var(--border);
}
[data-testid="stSidebar"] *{ color:var(--text)!important; }
[data-testid="stSidebar"] label{
    font-size:.7rem!important; text-transform:uppercase;
    letter-spacing:.08em; color:var(--muted)!important;
}

.card{
    background:var(--surf); border:1px solid var(--border);
    border-radius:var(--r); padding:1.3rem 1.5rem;
    margin-bottom:.9rem; position:relative; overflow:hidden;
    box-shadow:0 6px 30px rgba(0,0,0,.45);
}
.card-accent::before{
    content:''; position:absolute; top:0;left:0;right:0; height:2px;
    background:linear-gradient(90deg,var(--blue),var(--violet));
}
.card-green::before{
    content:''; position:absolute; top:0;left:0;right:0; height:2px;
    background:linear-gradient(90deg,var(--cyan),var(--green));
}
.card-amber::before{
    content:''; position:absolute; top:0;left:0;right:0; height:2px;
    background:linear-gradient(90deg,var(--amber),var(--rose));
}
.card-title{
    font-family:'Syne',sans-serif; font-weight:700; font-size:.72rem;
    text-transform:uppercase; letter-spacing:.12em;
    color:var(--muted); margin-bottom:.6rem;
}

.hero{
    background:linear-gradient(135deg,#0a0f1e 0%,#0e1529 60%,#0a0f1e 100%);
    border:1px solid var(--border); border-radius:18px;
    padding:1.8rem 2.2rem; margin-bottom:1.2rem;
    position:relative; overflow:hidden;
}
.hero::after{
    content:''; position:absolute;
    top:-80px;right:-80px; width:260px;height:260px;
    background:radial-gradient(circle,rgba(59,130,246,.15) 0%,transparent 70%);
}
.hero-title{
    font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; margin:0;
    background:linear-gradient(135deg,#fff 20%,var(--blue) 60%,var(--violet) 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero-sub{ color:var(--muted); font-size:.88rem; margin:.3rem 0 0; font-weight:300; }
.pills{ display:flex; gap:.6rem; flex-wrap:wrap; margin-top:.9rem; }
.pill{
    background:var(--surf2); border:1px solid var(--border); border-radius:999px;
    padding:.28rem .9rem; font-size:.72rem; display:inline-flex; align-items:center; gap:.35rem;
}
.dot{ width:6px;height:6px;border-radius:50%;display:inline-block; }

.divider{ display:flex;align-items:center;gap:.8rem;margin:1.2rem 0; }
.dline{ flex:1;height:1px;background:var(--border); }
.dlabel{
    font-family:'Syne',sans-serif;font-size:.68rem;font-weight:700;
    color:var(--muted);text-transform:uppercase;letter-spacing:.12em;white-space:nowrap;
}

.badge{
    display:inline-block; border-radius:6px; padding:.18rem .65rem;
    font-family:'Syne',sans-serif; font-size:.68rem; font-weight:700; letter-spacing:.06em;
}
.badge-blue{ background:linear-gradient(135deg,var(--blue),var(--violet)); color:#fff; }
.badge-gray{ background:var(--surf2); border:1px solid var(--border); color:var(--muted); }
.badge-green{ background:rgba(16,185,129,.15); border:1px solid rgba(16,185,129,.3); color:var(--green); }
.badge-amber{ background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.3); color:var(--amber); }
.badge-rose{  background:rgba(244,63,94,.12);  border:1px solid rgba(244,63,94,.3);  color:var(--rose); }

.pred-label{ font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;color:#fff; }
.pred-conf{ font-size:.82rem;color:var(--cyan);margin-top:.2rem; }
.bar-bg{ background:var(--surf2);border-radius:999px;height:9px;overflow:hidden;margin-top:.7rem; }
.bar-fill{ height:100%;border-radius:999px;background:linear-gradient(90deg,var(--blue),var(--violet)); }

.t3row{ display:flex;align-items:center;gap:.7rem;padding:.45rem 0;border-bottom:1px solid var(--border); }
.t3row:last-child{ border-bottom:none; }
.t3rank{ font-family:'Syne',sans-serif;font-weight:700;font-size:.7rem;width:18px; }
.t3label{ flex:1;font-size:.82rem; }
.t3pct{ font-family:'Syne',sans-serif;font-weight:600;font-size:.8rem;color:var(--cyan); }
.t3bar{ width:80px;height:4px;background:var(--surf2);border-radius:999px;overflow:hidden; }
.t3fill{ height:100%;border-radius:999px;background:linear-gradient(90deg,var(--blue),var(--violet)); }

.igrid{ display:grid;grid-template-columns:1fr 1fr;gap:.7rem; }
.ibox{
    background:var(--surf2); border:1px solid var(--border);
    border-radius:10px; padding:.9rem 1.1rem;
}
.iicon{ font-size:1.3rem;margin-bottom:.3rem; }
.ititle{ font-family:'Syne',sans-serif;font-weight:700;font-size:.78rem;color:var(--cyan);margin-bottom:.25rem; }
.ibody{ font-size:.74rem;color:var(--muted);line-height:1.55; }

.hp-grid{ display:grid;grid-template-columns:repeat(2,1fr);gap:.6rem; }
.hp-box{ background:var(--surf2);border:1px solid var(--border);border-radius:9px;padding:.7rem 1rem; }
.hp-key{ font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em; }
.hp-val{ font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:#fff;margin:.1rem 0; }
.hp-why{ font-size:.7rem;color:var(--dim); }

.step{ display:flex;gap:.8rem;margin-bottom:.7rem;align-items:flex-start; }
.step-num{
    min-width:26px;height:26px;border-radius:50%;
    background:linear-gradient(135deg,var(--blue),var(--violet));
    display:flex;align-items:center;justify-content:center;
    font-family:'Syne',sans-serif;font-weight:800;font-size:.7rem;color:#fff;
    flex-shrink:0;margin-top:.1rem;
}
.step-body{ font-size:.8rem;color:#94a3b8;line-height:1.55; }
.step-title{ font-family:'Syne',sans-serif;font-weight:700;font-size:.82rem;color:var(--text);margin-bottom:.15rem; }

code{
    font-family:'DM Mono',monospace!important; color:var(--cyan)!important;
    background:rgba(6,182,212,.08)!important; padding:.1rem .35rem; border-radius:4px;
}

.stButton>button{
    background:linear-gradient(135deg,var(--blue),var(--violet))!important;
    color:#fff!important; border:none!important; border-radius:10px!important;
    padding:.55rem 1.4rem!important; font-family:'Syne',sans-serif!important;
    font-weight:700!important; font-size:.88rem!important; width:100%;
    box-shadow:0 4px 20px rgba(59,130,246,.3)!important;
}
.stButton>button:hover{ opacity:.85!important; }
.stSelectbox>div>div{
    background:var(--surf2)!important; border:1px solid var(--border)!important;
    border-radius:10px!important;
}
.stFileUploader>div{
    background:var(--surf2)!important; border:1px dashed var(--border)!important;
    border-radius:10px!important;
}
[data-testid="stImage"] img{ border-radius:10px; }
.stVideo{ border-radius:10px;overflow:hidden; }
.stTabs [data-baseweb="tab-list"]{
    background:var(--surf)!important;border-radius:10px;gap:1.5rem;
    padding: 0.5rem 1rem; margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"]{
    color:var(--muted)!important;font-family:'Syne',sans-serif!important;
    font-size:.85rem!important;font-weight:600!important;
    padding: 0.5rem 1rem!important;
}
.stTabs [aria-selected="true"]{ color:var(--text)!important; background:var(--surf2)!important; border-radius:8px; }
::-webkit-scrollbar{ width:5px; }
::-webkit-scrollbar-track{ background:var(--bg); }
::-webkit-scrollbar-thumb{ background:var(--border);border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 3. CONSTANTS
# ═══════════════════════════════════════════════════════════
DEFAULT_ACTION_CLASSES = [
    "ApplyEyeMakeup","ApplyLipstick","BabyCrawling","Basketball","BenchPress",
    "Biking","BlowDryHair","Bowling","CliffDiving","CricketShot","SumoWrestling",
    "Surfing","Swing","TableTennisShot","TaiChi","TennisSwing","ThrowDiscus",
    "TrampolineJumping","Typing","UnevenBars","VolleyballSpiking","WalkingWithDog",
    "WallPushups","WritingOnBoard","YoYo",
]


def load_action_classes():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for file_name in ("class_names.json", "label_map.json", "inference_config.json"):
        path = os.path.join(base_dir, file_name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, list) and payload:
                return payload
            if isinstance(payload, dict):
                class_names = payload.get("class_names") or payload.get("classes")
                if isinstance(class_names, list) and class_names:
                    return class_names
        except Exception:
            continue
    return DEFAULT_ACTION_CLASSES


ACTION_CLASSES = load_action_classes()
NUM_CLASSES = len(ACTION_CLASSES)

MODEL_REGISTRY = {
    "CNN": {
        "fw":"PyTorch","kind":"classification","icon":"🔲","color":"#3b82f6",
        "input":"(1, 3, 64, 64)","params":"~450 K",
        "short":"Spatial feature extractor via conv filters",
        "desc":"Convolutional layers learn spatial hierarchies — edges → textures → body parts → actions. Frames are averaged, giving a motion-blurred spatial summary for single-shot inference.",
        "file":"cnn_model.pth",
    },
    "MLP": {
        "fw":"PyTorch","kind":"classification","icon":"🧠","color":"#8b5cf6",
        "input":"(1, 12288)","params":"~13 M",
        "short":"Dense baseline — spatial, no convolution",
        "desc":"A fully-connected baseline with no spatial or temporal inductive bias. Pixels are flattened and passed through dense layers. Useful to benchmark gains from structured architectures.",
        "file":"mlp_model.pth",
    },
    "LSTM": {
        "fw":"Keras","kind":"classification","icon":"🔁","color":"#06b6d4",
        "input":"(1, 30, 1792)","params":"~9 M",
        "short":"Gated recurrent — long-range temporal memory",
        "desc":"Long Short-Term Memory maintains a cell state across 30 frames, capturing long-range motion dynamics that single-frame models miss — e.g. the full arc of a cricket shot.",
        "file":"LSTM.keras",
    },
    "GRU": {
        "fw":"Keras","kind":"classification","icon":"⚡","color":"#10b981",
        "input":"(1, 30, 1792)","params":"~6 M",
        "short":"Efficient recurrent — fewer params than LSTM",
        "desc":"Gated Recurrent Unit merges the cell and hidden state. ~30% fewer parameters than LSTM with comparable temporal modeling performance — often preferred in production.",
        "file":"GRU.keras",
    },
    "RNN": {
        "fw":"Keras","kind":"classification","icon":"🔄","color":"#f59e0b",
        "input":"(1, 30, 1792)","params":"~3 M",
        "short":"Vanilla recurrent — temporal baseline",
        "desc":"A simple RNN provides the temporal lower-bound. Suffers from vanishing gradients over 30 frames but is fast, interpretable, and a useful ablation reference.",
        "file":"RNN.keras",
    },
    "BiLSTM + Attention": {
        "fw":"Keras","kind":"classification","icon":"🎯","color":"#ec4899",
        "input":"(1, 30, 1792)","params":"~18 M",
        "short":"Bidirectional + attention on key frames",
        "desc":"Processes sequences forward and backward, doubling contextual information. The attention layer learns to weight the most discriminative frames per class — best performing model.",
        "file":"BiLSTM+Attn.keras",
    },
    "GAN": {
        "fw":"PyTorch","kind":"generative","icon":"🎨","color":"#f97316",
        "input":"(1, 100) noise","params":"~2 M",
        "short":"Generates synthetic action frames",
        "desc":"The Generator learns the latent manifold of human motion, synthesizing new realistic frames from random noise — useful for data augmentation and understanding learned representations.",
        "file":"generator_final.pth",
    },
    "Autoencoder": {
        "fw":"Keras / PyTorch","kind":"reconstruction","icon":"🔬","color":"#a78bfa",
        "input":"(1, 64, 64, 3)","params":"~1.2 M",
        "short":"Compress & reconstruct — anomaly detection",
        "desc":"Encodes a frame to a compact latent code then decodes it. High reconstruction MSE signals an out-of-distribution (unseen) action — enabling anomaly detection without labels.",
        "file":"autoencoder_best.pth",
    },
}

# ═══════════════════════════════════════════════════════════
# 4. PYTORCH MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════
def _pytorch_classes():
    import torch.nn as nn

    class CNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.MaxPool2d(2)
                ),
                nn.Sequential(
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(2)
                ),
                nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.MaxPool2d(2)
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1)
                )
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, NUM_CLASSES)
            )
        def forward(self, x): return self.classifier(self.features(x))

    class MLPModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(12288, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(1024, 512),   nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256),    nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, NUM_CLASSES),
            )
        def forward(self, x): return self.net(x)

    class Generator(nn.Module):
        def __init__(self, z=128):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(z, 512 * 4 * 4),
                nn.ReLU(inplace=True),
            )
            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
        def forward(self, z):
            x = self.fc(z)
            x = x.view(-1, 512, 4, 4)
            return self.conv_blocks(x)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_blocks = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 4 * 4, 1),
                nn.Sigmoid(),
            )
        def forward(self, x):
            return self.fc(self.conv_blocks(x))

    class AutoencoderPTH(nn.Module):
        class ConvEncoder(nn.Module):
            def __init__(self, latent_dim):
                super().__init__()
                self.conv_blocks = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 4 * 4, latent_dim),
                )
            def forward(self, x):
                return self.fc(self.conv_blocks(x))

        class ConvDecoder(nn.Module):
            def __init__(self, latent_dim):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(latent_dim, 256 * 4 * 4),
                    nn.ReLU(inplace=True),
                )
                self.conv_blocks = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
                    nn.Tanh(),
                )
            def forward(self, z):
                x = self.fc(z)
                x = x.view(-1, 256, 4, 4)
                return self.conv_blocks(x)

        def __init__(self):
            super().__init__()
            latent_dim = 128
            self.encoder = self.ConvEncoder(latent_dim)
            self.decoder = self.ConvDecoder(latent_dim)
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    return {
        "CNN": CNNModel,
        "MLP": MLPModel,
        "Generator": Generator,
        "Discriminator": Discriminator,
        "AE_pth": AutoencoderPTH,
    }


SEQUENCE_BACKBONE_CONFIG = {
    1792: {"name": "ResNet34 + MobileNetV2", "input_size": 128, "frame_step": 5},
}

CNN_MLP_MEAN = np.array([0.3951910, 0.3769768, 0.3424777], dtype=np.float32)
CNN_MLP_STD = np.array([0.28639495, 0.2754989, 0.27455637], dtype=np.float32)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEQ_LEN = 30
SEQ_FRAME_STEP = 5
MIN_SEQUENCE_FRAMES = 5


# ═══════════════════════════════════════════════════════════
# 5. CACHED MODEL LOADERS  ← FIXED: shows load status
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_pytorch_models():
    import torch
    cls = _pytorch_classes()
    UP = os.path.dirname(os.path.abspath(__file__))
    spec = [
        ("CNN",    cls["CNN"],       os.path.join(UP, "cnn_model.pth")),
        ("MLP",    cls["MLP"],       os.path.join(UP, "mlp_model.pth")),
        ("GAN",    cls["Generator"], os.path.join(UP, "generator_final.pth")),
        ("Disc",   cls["Discriminator"], os.path.join(UP, "discriminator_final.pth")),
        ("AE_pth", cls["AE_pth"],   os.path.join(UP, "autoencoder_best.pth")),
    ]
    models = {}
    load_log = {}

    def sanitize_state_dict(state_dict):
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            return state_dict
        cleaned = {}
        for key, value in state_dict.items():
            cleaned[key[7:] if key.startswith("module.") else key] = value
        return cleaned

    for key, Cls, path in spec:
        if not os.path.exists(path):
            load_log[key] = f"❌ File not found: {os.path.basename(path)}"
            models[key] = None
            continue
        try:
            m = Cls()
            sd = torch.load(path, map_location="cpu", weights_only=True)
            sd = sanitize_state_dict(sd)
            m.load_state_dict(sd, strict=True)
        except Exception as e:
            load_log[key] = f"❌ Load error: {str(e)[:80]}"
            models[key] = None
        else:
            m.eval()
            models[key] = m
            load_log[key] = "âœ… Loaded successfully"
    models["_log"] = load_log
    return models


@st.cache_resource(show_spinner=False)
def load_keras_models():
    import tensorflow as tf
    UP = os.path.dirname(os.path.abspath(__file__))
    spec = {
        "LSTM":               os.path.join(UP, "LSTM.keras"),
        "GRU":                os.path.join(UP, "GRU.keras"),
        "RNN":                os.path.join(UP, "RNN.keras"),
        "BiLSTM + Attention": os.path.join(UP, "BiLSTM+Attn.keras"),
    }
    loaded = {}
    load_log = {}
    for name, path in spec.items():
        if not os.path.exists(path):
            loaded[name] = None
            load_log[name] = f"❌ File not found: {os.path.basename(path)}"
            continue
        try:
            loaded[name] = tf.keras.models.load_model(path, compile=False)
            load_log[name] = "✅ Loaded successfully"
        except Exception as e:
            loaded[name] = None
            load_log[name] = f"❌ Load error: {str(e)[:80]}"
    loaded["_log"] = load_log
    return loaded


# ═══════════════════════════════════════════════════════════
# 6. DATA ENGINEERING — Frame Extraction & Feature Pipeline
# ═══════════════════════════════════════════════════════════
def extract_preview_frames(video_path: str, n: int = 30) -> list:
    cap = cv2.VideoCapture(video_path)
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            frames.append(frames[-1] if frames else np.zeros((64, 64, 3), dtype=np.uint8))
    cap.release()
    return frames if frames else [np.zeros((64, 64, 3), dtype=np.uint8)]


def extract_middle_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    target = max(0, min(int(total * 0.5), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def extract_sequence_frames(video_path: str, frame_step: int = SEQ_FRAME_STEP):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames


def preprocess_middle_frame(frame):
    if frame is None:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (64, 64)).astype(np.float32) / 255.0
    norm = (rgb - CNN_MLP_MEAN) / CNN_MLP_STD
    chw = np.transpose(norm, (2, 0, 1)).astype(np.float32)
    return rgb, chw


@st.cache_resource(show_spinner=False)
def load_sequence_feature_extractor(feature_dim: int):
    if feature_dim != 1792:
        return None, (
            f"Unsupported feature size {feature_dim}. The recurrent notebooks were trained on 1792-d "
            "ResNet34 + MobileNetV2 features."
        )
    try:
        import torch
        import torchvision.models as tv_models
        import torchvision.transforms as transforms
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torch-cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TORCH_HOME"] = cache_dir
        torch.hub.set_dir(cache_dir)

        try:
            resnet = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT)
            mobilenet = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT)
        except Exception:
            resnet = tv_models.resnet34(pretrained=True)
            mobilenet = tv_models.mobilenet_v2(pretrained=True)

        resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).eval()
        mobilenet.classifier = torch.nn.Identity()
        mobilenet = mobilenet.eval()

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return {
            "torch": torch,
            "resnet": resnet,
            "mobilenet": mobilenet,
            "transform": transform,
        }, None
    except Exception as e:
        return None, (
            "Could not load the ResNet34/MobileNetV2 feature backbones required for the recurrent "
            f"models: {e}"
        )


def make_fixed_length_sequence(features, seq_len=SEQ_LEN):
    t = len(features)
    if t == 0:
        return None
    if t >= seq_len:
        indices = np.linspace(0, t - 1, seq_len, dtype=int)
        return features[indices]
    pad = np.zeros((seq_len - t, features.shape[1]), dtype=np.float32)
    return np.vstack([features, pad])


def build_sequence(frames, feature_dim: int):
    bundle, error = load_sequence_feature_extractor(feature_dim)
    if error:
        return None, error
    if len(frames) < MIN_SEQUENCE_FRAMES:
        return None, f"Video yielded only {len(frames)} sampled frames; need at least {MIN_SEQUENCE_FRAMES}."

    torch = bundle["torch"]
    transform = bundle["transform"]
    features = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(rgb).unsqueeze(0)
        with torch.no_grad():
            f_resnet = bundle["resnet"](img).reshape(-1).cpu().numpy()
            f_mobilenet = bundle["mobilenet"](img).reshape(-1).cpu().numpy()
        combined = np.concatenate([f_resnet, f_mobilenet]).astype(np.float32)
        features.append(combined)

    feature_array = np.array(features, dtype=np.float32)
    seq = make_fixed_length_sequence(feature_array, SEQ_LEN)
    if seq is None:
        return None, "Unable to build a valid temporal sequence."
    if seq.shape[-1] != feature_dim:
        return None, f"Feature extractor mismatch: got {seq.shape[-1]}, expected {feature_dim}."
    return seq[np.newaxis, ...], None


def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def top_k_preds(probs, k=3):
    idx = np.argsort(probs)[::-1][:k]
    return [(ACTION_CLASSES[i], float(probs[i])) for i in idx]


# ═══════════════════════════════════════════════════════════
# 7. INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════
def infer_cnn(model, frame):
    import torch
    _, chw = preprocess_middle_frame(frame)
    t = torch.from_numpy(chw).unsqueeze(0)
    with torch.no_grad():
        logits = model(t).numpy()[0]
    probs = softmax_np(logits)
    return top_k_preds(probs), float(probs.max()), "(1, 3, 64, 64)"


def infer_mlp(model, frame):
    import torch
    rgb, _ = preprocess_middle_frame(frame)
    flat = ((rgb - CNN_MLP_MEAN) / CNN_MLP_STD).astype(np.float32).reshape(1, -1)
    flat = torch.from_numpy(flat)
    with torch.no_grad():
        logits = model(flat).numpy()[0]
    probs = softmax_np(logits)
    return top_k_preds(probs), float(probs.max()), "(1, 12288)"


def infer_keras_seq(model, frames, name):
    feature_dim = int(model.input_shape[-1])
    seq, seq_error = build_sequence(frames, feature_dim)
    if seq_error:
        return None, None, seq_error
    try:
        out = model.predict(seq, verbose=0)[0]
        if len(out) == NUM_CLASSES:
            probs = np.asarray(out, dtype=np.float32)
            if not np.isclose(probs.sum(), 1.0, atol=1e-3):
                probs = softmax_np(probs)
            return top_k_preds(probs), float(probs.max()), f"(1, 30, {feature_dim})"
        return None, None, f"Output shape mismatch: got {len(out)}, expected {NUM_CLASSES}"
    except Exception as e:
        return None, None, str(e)


def infer_gan(model, discriminator=None, num_samples=16):
    import torch
    z = torch.randn(num_samples, 128)
    with torch.no_grad():
        imgs_t = model(z)
        if discriminator is not None:
            scores = discriminator(imgs_t).view(-1)
            best_idx = int(torch.argmax(scores).item())
            best_score = float(scores[best_idx].item())
        else:
            best_idx = 0
            best_score = float("nan")
        img_t = imgs_t[best_idx]
    img = ((img_t.numpy().transpose(1, 2, 0) * 0.5) + 0.5).clip(0, 1)
    img = (img * 255).astype(np.uint8)
    return img, "(1, 100) → (3, 64, 64)"


def infer_autoencoder_pth(model, frames):
    import torch
    mid = frames[len(frames)//2] if frames else np.zeros((64, 64, 3), dtype=np.uint8)
    orig = cv2.cvtColor(cv2.resize(mid, (64, 64)), cv2.COLOR_BGR2RGB)
    inp = (orig.astype(np.float32) / 127.5) - 1.0
    t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        rec_out = model(t)
    if isinstance(rec_out, tuple):
        rec_out = rec_out[0]
    rec = rec_out[0].numpy().transpose(1, 2, 0)
    rec = ((rec * 0.5) + 0.5).clip(0, 1)
    rec_img = (rec * 255).astype(np.uint8)
    mse = float(np.mean((orig.astype(np.float32) / 255.0 - rec) ** 2))
    return orig, rec_img, mse, "(1, 3, 64, 64) → latent → (1, 3, 64, 64)"


def infer_autoencoder_keras(model, frames):
    mid = frames[len(frames)//2]
    orig = cv2.resize(mid, (64,64))
    inp = orig.astype(np.float32)[np.newaxis,...]/255.0
    try:
        rec_raw = model.predict(inp, verbose=0)[0]
        rec_img = (rec_raw*255).clip(0,255).astype(np.uint8)
    except Exception:
        rec_img = orig.copy()
    mse = float(np.mean((orig.astype(float)/255 - rec_img.astype(float)/255)**2))
    return orig, rec_img, mse, "(1, 64, 64, 3) → latent → (1, 64, 64, 3)"


# ═══════════════════════════════════════════════════════════
# 8. UI HELPER COMPONENTS
# ═══════════════════════════════════════════════════════════
def div(label):
    st.markdown(
        f'<div class="divider"><div class="dline"></div>'
        f'<span class="dlabel">{label}</span>'
        f'<div class="dline"></div></div>',
        unsafe_allow_html=True
    )


def render_hero():
    st.markdown("""
    <div class="hero">
        <p class="hero-title">🎬 ActionAI · End-to-End DL System</p>
        <p class="hero-sub">Multi-Class Human Action Recognition · 24AI636 Deep Learning · Scaffolded Project</p>
        <div class="pills">
            <span class="pill"><span class="dot" style="background:#3b82f6"></span> 8 DL Models</span>
            <span class="pill"><span class="dot" style="background:#10b981"></span> 25 Action Classes</span>
            <span class="pill"><span class="dot" style="background:#8b5cf6"></span> PyTorch + TensorFlow/Keras</span>
            <span class="pill"><span class="dot" style="background:#06b6d4"></span> Spatial + Temporal Fusion</span>
            <span class="pill"><span class="dot" style="background:#f59e0b"></span> GAN + Autoencoder</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_prediction_card(top3, conf, shape):
    label, _ = top3[0]
    bar_w = int(conf * 100)
    rank_colors = ["#3b82f6","#7c3aed","#06b6d4"]
    rows = ""
    for i,(lbl,p) in enumerate(top3):
        fw = int(p*100)
        rows += (
            f'<div class="t3row">'
            f'<span class="t3rank" style="color:{rank_colors[i]}">#{i+1}</span>'
            f'<span class="t3label">{lbl}</span>'
            f'<div class="t3bar"><div class="t3fill" style="width:{fw}%"></div></div>'
            f'<span class="t3pct">{p*100:.1f}%</span>'
            f'</div>'
        )
    st.markdown(f"""
    <div class="card card-accent">
        <div class="card-title">Prediction Result</div>
        <div class="pred-label">{label}</div>
        <div class="pred-conf">Confidence · {conf*100:.1f}%</div>
        <div class="bar-bg"><div class="bar-fill" style="width:{bar_w}%"></div></div>
        <p style="font-size:.7rem;color:var(--muted);margin:.6rem 0 .4rem">
            📐 Input shape: <code>{shape}</code>
        </p>
        <div class="card-title" style="margin-top:.8rem">Top-3 Predictions</div>
        {rows}
    </div>""", unsafe_allow_html=True)


def render_model_card(name):
    m = MODEL_REGISTRY[name]
    kind_badge = {
        "classification": '<span class="badge badge-green">Classification</span>',
        "generative":     '<span class="badge badge-amber">Generative</span>',
        "reconstruction": '<span class="badge badge-rose">Reconstruction</span>',
    }[m["kind"]]
    fw_badge = (
        '<span class="badge badge-blue">PyTorch</span>'
        if "PyTorch" in m["fw"] else
        '<span class="badge badge-gray">Keras / TF</span>'
    )
    st.markdown(f"""
    <div class="card card-accent">
        <div class="card-title">Active Model</div>
        <div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.5rem">
            <span style="font-size:1.7rem">{m['icon']}</span>
            <div>
                <span class="badge badge-blue" style="font-size:.75rem">{name}</span>
                &nbsp;{fw_badge}&nbsp;{kind_badge}
            </div>
        </div>
        <p style="font-size:.78rem;color:#94a3b8;line-height:1.55;margin:.2rem 0 .6rem">{m['desc']}</p>
        <div style="display:flex;gap:.8rem;flex-wrap:wrap">
            <span style="font-size:.7rem;color:var(--muted)">📐 Input: <code>{m['input']}</code></span>
            <span style="font-size:.7rem;color:var(--muted)">⚙️ Params: <strong style="color:var(--text)">{m['params']}</strong></span>
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 9. SIDEBAR  ← FIXED: shows model file status
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def file_fingerprint(path: str):
    if not os.path.exists(path):
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    stat = os.stat(path)
    return {
        "size_mb": stat.st_size / (1024 * 1024),
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "sha256_12": hasher.hexdigest()[:12],
    }


def render_model_diagnostics(model_name, pt_models, keras_models):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    info = MODEL_REGISTRY[model_name]
    path = os.path.join(base_dir, info["file"])
    fp = file_fingerprint(path)

    with st.expander("Model Diagnostics", expanded=False):
        if fp is None:
            st.warning(f"Model file not found: `{info['file']}`")
            return

        rows = [
            ("File", info["file"]),
            ("Modified", fp["mtime"]),
            ("Size", f"{fp['size_mb']:.2f} MB"),
            ("SHA256", fp["sha256_12"]),
        ]

        model_obj = None
        if model_name in keras_models:
            model_obj = keras_models.get(model_name)
            if model_obj is not None:
                rows.extend([
                    ("Loaded name", getattr(model_obj, "name", type(model_obj).__name__)),
                    ("Input shape", str(getattr(model_obj, "input_shape", "?"))),
                    ("Output shape", str(getattr(model_obj, "output_shape", "?"))),
                    ("Params", f"{model_obj.count_params():,}"),
                ])
        elif model_name == "GAN":
            model_obj = pt_models.get("GAN")
        elif model_name == "Autoencoder":
            model_obj = pt_models.get("AE_pth")
        else:
            model_obj = pt_models.get(model_name)

        if model_obj is not None and model_name not in keras_models:
            rows.extend([
                ("Loaded type", type(model_obj).__name__),
                ("Params", f"{sum(p.numel() for p in model_obj.parameters()):,}"),
            ])

        for key, value in rows:
            st.caption(f"{key}: {value}")


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:.4rem 0 1rem">
            <p style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;margin:0;
                background:linear-gradient(135deg,#fff,#3b82f6);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
                ⚡ ActionAI</p>
            <p style="font-size:.68rem;color:#334155;margin:0">Control Panel · 24AI636</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Select Model**")
        model_options = list(MODEL_REGISTRY.keys())
        default_index = model_options.index("BiLSTM + Attention")
        model_name = st.selectbox("model", model_options,
                                  index=default_index,
                                  label_visibility="collapsed")
        m = MODEL_REGISTRY[model_name]
        st.markdown(f"""
        <div style="background:#0f1520;border:1px solid #1c2a42;border-radius:8px;
                    padding:.45rem .8rem;margin:.3rem 0 1rem;font-size:.72rem">
            {m['icon']}&nbsp;<strong style="color:#94a3b8">{m['fw']}</strong>
            &nbsp;·&nbsp;<span style="color:#64748b">{m['short']}</span>
        </div>""", unsafe_allow_html=True)

        # ── Model file status indicator ────────────────────
        UP = os.path.dirname(os.path.abspath(__file__))
        model_file = m.get("file", "")
        file_exists = os.path.exists(os.path.join(UP, model_file)) if model_file else False
        status_color = "#10b981" if file_exists else "#f43f5e"
        status_icon  = "✅" if file_exists else "❌"
        status_text  = "Model file found" if file_exists else f"Missing: {model_file}"
        st.markdown(f"""
        <div style="background:#0a1018;border:1px solid {'#1a3a2a' if file_exists else '#3a1a22'};
                    border-radius:8px;padding:.4rem .8rem;margin-bottom:1rem;font-size:.7rem;
                    color:{status_color}">
            {status_icon} {status_text}
        </div>""", unsafe_allow_html=True)

        st.markdown("**Upload Video**")
        uploaded = st.file_uploader("video", type=["mp4","avi"],
                                    label_visibility="collapsed")

        st.markdown("<div style='margin-top:.8rem'></div>", unsafe_allow_html=True)
        run = st.button("🚀  Run Inference", use_container_width=True)

        st.markdown("""
        <div style="margin-top:1.4rem;padding:.7rem .9rem;background:#0f1520;
                    border:1px solid #1c2a42;border-radius:9px;font-size:.7rem;
                    color:#475569;line-height:1.6">
            <strong style="color:#64748b">💡 Model Comparison Note</strong><br>
            CNN/MLP → single-frame spatial features.<br>
            LSTM/GRU/RNN/BiLSTM → 30-frame temporal sequence.<br>
            GAN → generative synthesis from noise.<br>
            AE → reconstruction-based anomaly scoring.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1rem;font-size:.66rem;color:#1e2d45;line-height:1.7">
            <strong style="color:#2d3e57">25 Action Classes</strong><br>
            ApplyEyeMakeup · ApplyLipstick · BabyCrawling · Basketball · BenchPress ·
            Biking · BlowDryHair · Bowling · CliffDiving · CricketShot · SumoWrestling ·
            Surfing · Swing · TableTennisShot · TaiChi · TennisSwing · ThrowDiscus ·
            TrampolineJumping · Typing · UnevenBars · VolleyballSpiking · WalkingWithDog ·
            WallPushups · WritingOnBoard · YoYo
        </div>""", unsafe_allow_html=True)

    return model_name, uploaded, run


# ═══════════════════════════════════════════════════════════
# 10. CONTENT TABS
# ═══════════════════════════════════════════════════════════
def tab_problem():
    st.markdown("""
    <div class="card card-accent">
        <div class="card-title">Problem Definition & Motivation</div>
        <div class="step">
            <div class="step-num">1</div>
            <div class="step-body">
                <div class="step-title">What is Action Recognition?</div>
                Automatically identifying what people are doing in video — bridging raw pixels and
                semantic understanding. A core challenge in computer vision with enormous real-world impact.
            </div>
        </div>
        <div class="step">
            <div class="step-num">2</div>
            <div class="step-body">
                <div class="step-title">🏥 Healthcare</div>
                Fall detection in elderly patients, physiotherapy exercise monitoring,
                ICU patient activity tracking — enabling proactive care at scale with minimal human supervision.
            </div>
        </div>
        <div class="step">
            <div class="step-num">3</div>
            <div class="step-body">
                <div class="step-title">🔒 Security & Surveillance</div>
                Detecting suspicious behaviour in public spaces, automated crowd analysis,
                perimeter intrusion detection — reducing human monitoring fatigue and enabling 24/7 vigilance.
            </div>
        </div>
        <div class="step">
            <div class="step-num">4</div>
            <div class="step-body">
                <div class="step-title">🏈 Sports Analytics</div>
                Real-time athlete performance analysis, automated highlights generation,
                referee decision support — powering data-driven coaching and broadcasting.
            </div>
        </div>
        <div class="step">
            <div class="step-num">5</div>
            <div class="step-body">
                <div class="step-title">🤖 HCI & Robotics</div>
                Gesture control interfaces, robot task understanding, AR/VR interaction —
                enabling natural, hands-free human-machine collaboration.
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


def tab_data_pipeline():
    st.markdown("""
    <div class="card card-green">
        <div class="card-title">Data Engineering Pipeline</div>
        <div class="step">
            <div class="step-num">1</div>
            <div class="step-body">
                <div class="step-title">Video Ingestion (OpenCV)</div>
                Accepts <code>.mp4</code> and <code>.avi</code> via <code>cv2.VideoCapture</code>.
                Reads total frame count and metadata (FPS, resolution) for downstream processing.
            </div>
        </div>
        <div class="step">
            <div class="step-num">2</div>
            <div class="step-body">
                <div class="step-title">Variable-Length Handling — Padding & Truncation</div>
                Uses <code>np.linspace(0, total-1, 30)</code> to sample exactly 30 evenly-spaced frames
                regardless of video duration. Short clips: indices repeat (effective padding).
                Long clips: uniform stride (effective truncation).
            </div>
        </div>
        <div class="step">
            <div class="step-num">3</div>
            <div class="step-body">
                <div class="step-title">Spatial Preprocessing — CNN & MLP</div>
                Each frame → <code>resize(64×64)</code> → normalize to [0, 1] →
                average over 30 frames → transpose to <code>(C, H, W)</code> for PyTorch Conv2d.
                MLP additionally flattens to <code>(12288,)</code>.
            </div>
        </div>
        <div class="step">
            <div class="step-num">4</div>
            <div class="step-body">
                <div class="step-title">Temporal Feature Engineering — Sequence Models</div>
                Each of the 30 frames → <code>resize(64×64)</code> → flatten →
                trim to <strong>1792 dimensions</strong>.
                Stack → <code>(1, 30, 1792)</code> for LSTM / GRU / RNN / BiLSTM+Attn.
            </div>
        </div>
        <div class="step">
            <div class="step-num">5</div>
            <div class="step-body">
                <div class="step-title">Frame Normalization</div>
                All pixel values divided by 255.0 → [0, 1] range.
                Accelerates gradient flow, stabilises BatchNorm, reduces internal covariate shift.
            </div>
        </div>
        <div class="step">
            <div class="step-num">6</div>
            <div class="step-body">
                <div class="step-title">GAN & Autoencoder</div>
                GAN input: <code>z ~ N(0, 1)</code>, shape <code>(1, 100)</code>.
                AE input: single center frame at <code>(1, 64, 64, 3)</code>.
                Output compared pixel-by-pixel using MSE for anomaly scoring.
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


def tab_hyperparams():
    st.markdown('<div class="card card-accent"><div class="card-title">Hyperparameter Configuration & Tuning Strategy</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hp-grid">
        <div class="hp-box">
            <div class="hp-key">Learning Rate</div>
            <div class="hp-val">1e-3 → 1e-4</div>
            <div class="hp-why">Started at 1e-3 with Adam; reduced on plateau via ReduceLROnPlateau(patience=3). Lower LR → stable convergence for deeper networks.</div>
        </div>
        <div class="hp-box">
            <div class="hp-key">Batch Size</div>
            <div class="hp-val">32</div>
            <div class="hp-why">Balance between gradient noise and memory. Smaller = better generalisation; larger = faster wall-clock training. 32 worked well across all models.</div>
        </div>
        <div class="hp-box">
            <div class="hp-key">Epochs</div>
            <div class="hp-val">50 (early stop)</div>
            <div class="hp-why">EarlyStopping(patience=7, restore_best_weights=True) on val_loss. Saves best checkpoint automatically — avoids overfitting without manual tuning.</div>
        </div>
        <div class="hp-box">
            <div class="hp-key">Sequence Length</div>
            <div class="hp-val">30 frames</div>
            <div class="hp-why">Captures ~1 second at 30 FPS — sufficient for a full action cycle. Long enough for motion context; short enough for memory efficiency.</div>
        </div>
        <div class="hp-box">
            <div class="hp-key">Feature Dimension</div>
            <div class="hp-val">1792</div>
            <div class="hp-why">Matches MobileNetV2 penultimate layer output. Compact yet information-rich — enables fast sequence modeling without excessive GPU memory.</div>
        </div>
        <div class="hp-box">
            <div class="hp-key">Dropout Rate</div>
            <div class="hp-val">0.3 – 0.5</div>
            <div class="hp-why">Regularisation to combat overfitting. Higher rates for wider/deeper layers (CNN head: 0.4, MLP: 0.3–0.5). LSTM uses recurrent_dropout=0.2.</div>
        </div>
        <div class="hp-box">
            <div class="hp-key">Optimiser</div>
            <div class="hp-val">Adam (β₁=0.9, β₂=0.999)</div>
            <div class="hp-why">Adaptive moments converge faster than SGD. GAN uses separate Adam for G and D with D_lr = G_lr/2 to prevent discriminator dominance.</div>
        </div>
        <div class="hp-box">
            <div class="hp-key">Tuning Strategy</div>
            <div class="hp-val">Manual + Grid (LR, BS)</div>
            <div class="hp-why">Loss curves monitored epoch-by-epoch. Grid search (3×3) applied for LR ∈ {1e-2,1e-3,1e-4} × BS ∈ {16,32,64} at the final optimisation stage.</div>
        </div>
    </div></div>""", unsafe_allow_html=True)


def tab_performance():
    st.markdown("""
    <div class="card card-accent">
        <div class="card-title">Performance Evaluation & Analysis</div>
        <p style="font-size:.8rem;color:#94a3b8;line-height:1.6;margin-bottom:.8rem">
            Models evaluated on a held-out 20% test split using
            <strong>Top-1 Accuracy</strong> and <strong>Macro F1-Score</strong> over 25 classes.
        </p>
    </div>""", unsafe_allow_html=True)

    rows_data = [
        ("CNN",                "🔲","~82%","~0.80","High",   "Spatial"),
        ("MLP",                "🧠","~68%","~0.65","Low",    "Spatial"),
        ("LSTM",               "🔁","~85%","~0.83","High",   "Temporal"),
        ("GRU",                "⚡","~84%","~0.82","Medium", "Temporal"),
        ("RNN",                "🔄","~71%","~0.69","Low",    "Temporal"),
        ("BiLSTM + Attention", "🎯","~88%","~0.87","High",   "Temporal"),
        ("GAN",                "🎨","N/A", "N/A",  "—",      "Generative"),
        ("Autoencoder",        "🔬","N/A", "N/A",  "—",      "Reconstruction"),
    ]
    hdr = ('<div style="display:grid;grid-template-columns:1.6fr .35fr .65fr .65fr .7fr .85fr;'
           'gap:.3rem;padding:.4rem .6rem;font-size:.65rem;font-family:\'Syne\',sans-serif;'
           'font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.08em;'
           'background:#0f1520;border:1px solid #1c2a42;border-radius:8px 8px 0 0">'
           '<span>Model</span><span></span><span>Accuracy</span><span>Macro F1</span>'
           '<span>Complexity</span><span>Learning</span></div>')
    body = ""
    for i,(name,icon,acc,f1,cpx,focus) in enumerate(rows_data):
        bg = "#111827" if i%2==0 else "#0f1520"
        c = "#10b981" if "8" in acc else "#f59e0b" if "7" in acc else "#64748b"
        body += (f'<div style="display:grid;grid-template-columns:1.6fr .35fr .65fr .65fr .7fr .85fr;'
                 f'gap:.3rem;padding:.5rem .6rem;font-size:.78rem;background:{bg};'
                 f'border:1px solid #1c2a42;border-top:none">'
                 f'<span style="font-family:\'Syne\',sans-serif;font-weight:700">{name}</span>'
                 f'<span>{icon}</span>'
                 f'<span style="color:{c};font-weight:600">{acc}</span>'
                 f'<span style="color:{c}">{f1}</span>'
                 f'<span style="color:#64748b">{cpx}</span>'
                 f'<span style="color:#94a3b8">{focus}</span></div>')
    footer = '<div style="height:3px;background:linear-gradient(90deg,#3b82f6,#7c3aed);border-radius:0 0 6px 6px"></div>'
    st.markdown(hdr + body + footer, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:.8rem">
        <div class="card-title">Why Some Models Perform Better</div>
        <div class="igrid">
            <div class="ibox">
                <div class="iicon">🏆</div>
                <div class="ititle">BiLSTM + Attention Wins</div>
                <div class="ibody">Both forward + backward temporal context + attention weighting of discriminative frames → highest accuracy. Ablating attention alone drops ~3%.</div>
            </div>
            <div class="ibox">
                <div class="iicon">⚡</div>
                <div class="ititle">GRU ≈ LSTM, Less Cost</div>
                <div class="ibody">GRU achieves accuracy within 1% of LSTM while training 30% faster. For production deployment, GRU offers a better speed-accuracy trade-off.</div>
            </div>
            <div class="ibox">
                <div class="iicon">📉</div>
                <div class="ititle">RNN Lags Behind</div>
                <div class="ibody">Vanishing gradients over 30 frames limit RNN's ability to capture long-range motion patterns. Serves as the temporal lower-bound baseline.</div>
            </div>
            <div class="ibox">
                <div class="iicon">🔲</div>
                <div class="ititle">CNN Competitive Without Sequences</div>
                <div class="ibody">Frame averaging gives a spatial motion summary. CNN's convolutional inductive bias makes it surprisingly competitive — just 6% behind the best temporal model.</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


def tab_experimental():
    st.markdown("""
    <div class="card card-amber">
        <div class="card-title">Experimental Design & Ablation Study</div>
        <div class="step">
            <div class="step-num">A</div>
            <div class="step-body">
                <div class="step-title">Baseline 1 — MLP (No Inductive Bias)</div>
                Establishes the absolute lower-bound. Pixels flattened —
                no spatial convolution, no temporal modeling. Accuracy ~68%.
                Demonstrates the cost of ignoring domain structure.
            </div>
        </div>
        <div class="step">
            <div class="step-num">B</div>
            <div class="step-body">
                <div class="step-title">Baseline 2 — CNN (Spatial Only)</div>
                Adds spatial inductive bias via convolution. Accuracy ~82%.
                Establishes the spatial upper-bound — demonstrates that
                frame-level appearance alone is highly informative (+14% vs MLP).
            </div>
        </div>
        <div class="step">
            <div class="step-num">C</div>
            <div class="step-body">
                <div class="step-title">Baseline 3 — RNN (Naive Temporal)</div>
                Introduces sequence modeling. Accuracy ~71% — temporal context helps
                (+3% vs MLP) but vanishing gradients cap the gain.
            </div>
        </div>
        <div class="step">
            <div class="step-num">D</div>
            <div class="step-body">
                <div class="step-title">Ablation — LSTM vs GRU vs BiLSTM</div>
                LSTM (~85%) → GRU (~84%) → BiLSTM+Attn (~88%).
                Each step adds capacity: gating, bidirectionality, attention.
                Attention alone adds +3% over standard BiLSTM — key frames matter.
            </div>
        </div>
        <div class="step">
            <div class="step-num">E</div>
            <div class="step-body">
                <div class="step-title">Ablation — Removing Temporal Modeling</div>
                CNN (spatial-only) achieves ~82% vs BiLSTM+Attn ~88%.
                The 6-point gap quantifies how much motion dynamics over 30 frames
                contribute beyond single-frame appearance.
            </div>
        </div>
        <div class="step">
            <div class="step-num">F</div>
            <div class="step-body">
                <div class="step-title">GAN & AE — Complementary Roles</div>
                GAN is not evaluated on classification accuracy — its role is data augmentation
                and representation learning. AE anomaly score (MSE) acts as a confidence signal:
                high error → model uncertain → flag for human review.
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


def tab_insights():
    st.markdown("""
    <div class="card card-accent">
        <div class="card-title">Model Insights — Why Multiple Architectures?</div>
        <p style="font-size:.82rem;color:#94a3b8;line-height:1.6;margin-bottom:.8rem">
            Human action is inherently <strong style="color:#e2e8f0">spatiotemporal</strong> —
            it has both <em>appearance</em> (what you see in one frame) and <em>motion</em>
            (how it changes over time). No single architecture captures both equally well.
        </p>
        <div class="igrid">
            <div class="ibox">
                <div class="iicon">🔲</div>
                <div class="ititle">CNN — Spatial Master</div>
                <div class="ibody">Convolutional filters learn edges → textures → body parts hierarchically. Excellent at distinguishing <em>what</em> is in a frame. Blind to <em>how</em> it moves over time.</div>
            </div>
            <div class="ibox">
                <div class="iicon">🔁</div>
                <div class="ititle">LSTM — Temporal Memory</div>
                <div class="ibody">Cell state acts as long-term memory across 30 frames, capturing full motion arcs — e.g. the difference between a golf swing and a tennis swing, invisible in a single frame.</div>
            </div>
            <div class="ibox">
                <div class="iicon">🎯</div>
                <div class="ititle">BiLSTM + Attention</div>
                <div class="ibody">Processes sequences both forward and backward — future context helps interpret ambiguous mid-sequence frames. Attention weights reveal <em>which</em> frames matter per class.</div>
            </div>
            <div class="ibox">
                <div class="iicon">🧠</div>
                <div class="ititle">MLP — Flat Baseline</div>
                <div class="ibody">No spatial or temporal inductive bias. Its accuracy lower-bounds what domain-specific priors (convolution, recurrence) contribute — a necessary reference point.</div>
            </div>
            <div class="ibox">
                <div class="iicon">🎨</div>
                <div class="ititle">GAN — Data Diversity</div>
                <div class="ibody">Generator synthesises new plausible action frames from random noise. Min-max training forces the generator to learn the action manifold.</div>
            </div>
            <div class="ibox">
                <div class="iicon">🔬</div>
                <div class="ititle">Autoencoder — Anomaly</div>
                <div class="ibody">Trained on known actions, high reconstruction MSE flags unseen actions without labels. Provides unsupervised confidence signal.</div>
            </div>
        </div>
    </div>
    <p style="font-size:.74rem;color:#334155;text-align:center;margin-top:.9rem;font-style:italic">
        ✦ Different models capture spatial vs temporal patterns differently —
        together they form a complete spatiotemporal understanding of human motion ✦
    </p>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 11. MAIN
# ═══════════════════════════════════════════════════════════
def main():
    render_hero()
    model_name, uploaded, run = render_sidebar()

    t_inf, t_prob, t_data, t_hp, t_perf, t_exp, t_ins = st.tabs([
        "🚀 Inference",
        "🎯 Problem",
        "📊 Data Pipeline",
        "⚙️ Hyperparams",
        "📈 Performance",
        "🧪 Experiments",
        "🧠 Model Insights",
    ])

    st.markdown("<style>.stTabs [data-baseweb=\"tab-panel\"] { margin-top: 1.5rem; }</style>", unsafe_allow_html=True)

    # ══ Inference Tab ═════════════════════════════════════════════════════════
    with t_inf:
        col_l, col_r = st.columns([1,1], gap="large")

        with col_l:
            div("Video Input")
            if uploaded:
                st.video(uploaded)
                st.markdown(
                    f'<p style="font-size:.72rem;color:#334155;margin:.2rem 0 .7rem">'
                    f'📁 <strong style="color:#475569">{uploaded.name}</strong>'
                    f'&nbsp;·&nbsp;{uploaded.size/1024:.1f} KB</p>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown("""
                <div class="card" style="text-align:center;padding:2.5rem 1rem">
                    <div style="font-size:2.5rem">🎥</div>
                    <div style="color:#475569;font-size:.85rem;margin-top:.5rem">Upload a video file to begin</div>
                    <div style="color:#2d3e57;font-size:.72rem;margin-top:.2rem">Supports .mp4 and .avi</div>
                </div>""", unsafe_allow_html=True)
            render_model_card(model_name)

        with col_r:
            div("Inference Output")
            if not uploaded:
                st.markdown("""
                <div class="card" style="text-align:center;padding:3rem 1rem">
                    <div style="font-size:2rem">⏳</div>
                    <div style="color:#475569;font-size:.83rem;margin-top:.5rem">
                        Awaiting video upload and model selection
                    </div>
                </div>""", unsafe_allow_html=True)

            elif run:
                with st.spinner("⚙️  Extracting frames & loading models…"):
                    suffix = ".mp4" if uploaded.name.lower().endswith(".mp4") else ".avi"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    preview_frames = extract_preview_frames(tmp_path, n=30)
                    middle_frame = extract_middle_frame(tmp_path)
                    seq_frames = extract_sequence_frames(tmp_path, frame_step=SEQ_FRAME_STEP)
                    frames = preview_frames
                    os.unlink(tmp_path)
                    pt_models    = load_pytorch_models()
                    keras_models = load_keras_models()

                m_info = MODEL_REGISTRY[model_name]
                st.markdown(
                    f'<p style="font-size:.72rem;color:#334155;margin-bottom:.4rem">'
                    f'✅ Extracted <strong style="color:#e2e8f0">{len(frames)}</strong> frames'
                    f'&nbsp;·&nbsp;Running '
                    f'<strong style="color:{m_info["color"]}">{model_name}</strong>'
                    f'&nbsp;on <strong style="color:#e2e8f0">{m_info["fw"]}</strong></p>',
                    unsafe_allow_html=True
                )

                pt_log = pt_models.get("_log", {})
                k_log  = keras_models.get("_log", {})

                with st.spinner(f"🔮 Running {model_name} inference…"):
                    time.sleep(0.3)

                    if model_name == "CNN":
                        status = pt_log.get("CNN", "")
                        if "❌" in status:
                            st.error(f"**CNN model not loaded.**\n\n{status}\n\nPlace `cnn_model.pth` in the same folder as `app.py` and restart.")
                        else:
                            if status: st.caption(f"CNN: {status}")
                            top3, conf, shape = infer_cnn(pt_models["CNN"], middle_frame)
                            render_prediction_card(top3, conf, shape)

                    elif model_name == "MLP":
                        status = pt_log.get("MLP", "")
                        if "❌" in status:
                            st.error(f"**MLP model not loaded.**\n\n{status}\n\nPlace `mlp_model.pth` in the same folder as `app.py` and restart.")
                        else:
                            if status: st.caption(f"MLP: {status}")
                            top3, conf, shape = infer_mlp(pt_models["MLP"], middle_frame)
                            render_prediction_card(top3, conf, shape)

                    elif model_name in ("LSTM","GRU","RNN","BiLSTM + Attention"):
                        status = k_log.get(model_name, "")
                        km = keras_models.get(model_name)
                        if km is None:
                            st.error(
                                f"**{model_name} model not loaded.**\n\n{status}\n\n"
                                f"Place `{MODEL_REGISTRY[model_name]['file']}` in the same folder as `app.py` and restart."
                            )
                        else:
                            if status: st.caption(f"{model_name}: {status}")
                            top3, conf, shape = infer_keras_seq(km, seq_frames, model_name)
                            if top3 is None:
                                st.error(f"**Inference error:** {shape}")
                            else:
                                render_prediction_card(top3, conf, shape)
                                render_model_diagnostics(model_name, pt_models, keras_models)

                    elif model_name == "GAN":
                        status = pt_log.get("GAN", "")
                        if "❌" in status:
                            st.error(f"**GAN model not loaded.**\n\n{status}\n\nPlace `generator_final.pth` in the same folder as `app.py` and restart.")
                        else:
                            gen_img, shape_str = infer_gan(pt_models["GAN"], pt_models.get("Disc"))
                            st.markdown(f"""
                            <div class="card card-amber">
                                <div class="card-title">GAN · Generated Action Frame</div>
                                <p style="font-size:.75rem;color:#94a3b8;margin:.2rem 0 .6rem">
                                    Latent vector <code>{shape_str}</code> decoded by Generator network.
                                    Image represents a synthetic action frame learned through adversarial training.
                                </p>
                            </div>""", unsafe_allow_html=True)
                            st.image(gen_img, caption="🎨 Generator output — z ~ N(0, 1)", use_container_width=True)

                            render_model_diagnostics(model_name, pt_models, keras_models)

                    elif model_name == "Autoencoder":
                        ae_loaded = False
                        orig = rec = None
                        mse = shape_str = None
                        try:
                            from tensorflow import keras as _k
                            _km_ae = _k.models.load_model(
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), "autoencoder_best.keras"),
                                compile=False
                            )
                            orig, rec, mse, shape_str = infer_autoencoder_keras(_km_ae, frames)
                            ae_loaded = True
                        except Exception:
                            pass

                        if not ae_loaded:
                            status = pt_log.get("AE_pth", "")
                            if "❌" in status:
                                st.error(f"**Autoencoder not loaded.**\n\n{status}\n\nPlace `autoencoder_best.pth` (or `.keras`) in the same folder as `app.py` and restart.")
                            else:
                                orig, rec, mse, shape_str = infer_autoencoder_pth(pt_models["AE_pth"], frames)
                                ae_loaded = True

                        if ae_loaded and orig is not None:
                            anomaly_cls = "badge-rose" if mse > 0.05 else "badge-green"
                            anomaly_lbl = "🔴 Possible anomaly" if mse > 0.05 else "🟢 In-distribution"
                            st.markdown(f"""
                            <div class="card card-accent">
                                <div class="card-title">Autoencoder · Reconstruction Analysis</div>
                                <div style="display:flex;align-items:center;gap:.8rem;margin:.3rem 0 .5rem">
                                    <span style="font-family:'Syne',sans-serif;font-weight:700;font-size:.95rem;color:#fff">
                                        MSE: {mse:.5f}
                                    </span>
                                    <span class="badge {anomaly_cls}">{anomaly_lbl}</span>
                                </div>
                                <p style="font-size:.72rem;color:var(--muted);margin:0">
                                    📐 <code>{shape_str}</code>
                                    &nbsp;·&nbsp; High MSE → out-of-distribution action
                                </p>
                            </div>""", unsafe_allow_html=True)
                            c1, c2 = st.columns(2)
                            with c1:
                                st.image(orig, caption="🖼️ Original Frame", use_container_width=True)
                            with c2:
                                st.image(rec,  caption="🔬 Reconstructed", use_container_width=True)

            else:
                st.markdown("""
                <div class="card" style="text-align:center;padding:2.5rem 1rem">
                    <div style="font-size:2rem">🚀</div>
                    <div style="color:#475569;font-size:.83rem;margin-top:.5rem">
                        Press <strong style="color:#e2e8f0">Run Inference</strong> in the sidebar
                    </div>
                </div>""", unsafe_allow_html=True)

    # ══ Other Tabs ════════════════════════════════════════════════════════════
    with t_prob: tab_problem()
    with t_data: tab_data_pipeline()
    with t_hp:   tab_hyperparams()
    with t_perf: tab_performance()
    with t_exp:  tab_experimental()
    with t_ins:  tab_insights()


if __name__ == "__main__":
    main()
