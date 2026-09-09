"""
Turning a pile of answers into a working copy.

Note there are no true labels anywhere in this file. The clone never sees
CIFAR-10's ground truth. Its entire education is "when the victim was shown
this picture, it said this." That is what makes it a stolen model rather than
a separately trained one -- and it's why fidelity, not accuracy, is the number
that proves theft.

This file is importable on purpose. Notebooks call train_clone(); they never
hold logic. (See the repo guide.)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .service import NORM_RECIPES


class StolenDataset(Dataset):
    """Images plus the victim's probability vectors, with light augmentation.

    Random crops and flips matter more than usual here: the thief has a small
    fixed set of answers and needs to squeeze every drop out of them.
    """

    def __init__(self, images, probs, norm_name="dataset_stats", augment=True):
        self.images = np.asarray(images, dtype=np.uint8)
        self.probs = np.asarray(probs, dtype=np.float32)
        self.mean, self.std = NORM_RECIPES[norm_name]
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :]
            pad = np.pad(img, ((4, 4), (4, 4), (0, 0)), mode="reflect")
            y, x = np.random.randint(0, 9, size=2)
            img = pad[y:y + 32, x:x + 32, :]
        x = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0).permute(2, 0, 1)
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        return (x - mean) / std, torch.from_numpy(self.probs[i])


def build_student(arch="cifar10_resnet20"):
    """Fresh weights, no pretraining. The copy starts knowing nothing.

    Swap arch (e.g. 'cifar10_mobilenetv2_x0_5', 'cifar10_vgg11_bn') to show the
    thief doesn't need to know how the victim was built.
    """
    return torch.hub.load("chenyaofo/pytorch-cifar-models", arch, pretrained=False)


def soft_label_loss(student_logits, victim_probs, temperature=1.0):
    """Cross-entropy against the victim's full distribution.

    With temperature=1 this is exactly 'match the victim's probabilities'.
    Raising it above 1 makes the clone pay more attention to the small
    probabilities -- the near-misses, which carry a lot of the boundary info.
    """
    if temperature != 1.0:
        victim_probs = victim_probs.pow(1.0 / temperature)
        victim_probs = victim_probs / victim_probs.sum(dim=1, keepdim=True)
    return -(victim_probs * F.log_softmax(student_logits, dim=1)).sum(dim=1).mean()


def train_clone(images, probs, arch="cifar10_resnet20", epochs=60, batch_size=128,
                lr=0.1, temperature=1.0, norm_name="dataset_stats", seed=0,
                device=None, verbose=True):
    """Train a copy from stolen query/response pairs. Returns the model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = StolenDataset(images, probs, norm_name=norm_name, augment=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                    drop_last=len(ds) > batch_size)

    student = build_student(arch).to(device).train()
    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=max(1, len(dl))
    )

    for ep in range(epochs):
        total, n = 0.0, 0
        for xb, pb in dl:
            xb, pb = xb.to(device), pb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = soft_label_loss(student(xb), pb, temperature)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(xb)
            n += len(xb)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    epoch {ep + 1:3d}/{epochs}  loss {total / max(n, 1):.4f}")

    return student.eval()
