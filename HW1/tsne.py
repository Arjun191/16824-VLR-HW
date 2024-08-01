import torch
from sklearn.manifold import TSNE
from torchvision.models.feature_extraction import create_feature_extractor
from utils import *
import matplotlib.pyplot as plt
from matplotlib import patches
from train_q2 import ResNet

model_path = "checkpoint-model-epoch50.pth"
model = ResNet(20).to('cuda')
model = torch.load(model_path)
model.eval()

test_loader = get_data_loader('voc', train=False, batch_size=100, split='test')

return_nodes = {'resnet': 'avgpool'}

truncated_model = create_feature_extractor(model, return_nodes=return_nodes)
feats = []
targets = []
n = 0

for data, target, _ in test_loader:
    feat = truncated_model(data.to("cuda"))['avgpool'].view((data.shape[0], -1))
    feats.append(feat.detach().cpu().numpy())
    targets.append(target.view(-1, 20).detach().cpu().numpy())
    n += 1
    if n == 10:
        break

feats = np.concatenate(feats)
targets = np.concatenate(targets).astype(np.int32)

tsne = TSNE()
proj = tsne.fit_transform(feats)

colors = np.array([[np.random.choice(np.arange(256), size=3)] for i in range(20)])
mean_colors = []
for i in range(proj.shape[0]):
    colors1 = colors[np.where(targets[i, :]==1)]
    mean_colors.append(np.mean(colors1, axis=0, dtype=np.int32))

plt.figure(figsize=(12,10))
plt.scatter(proj[:, 0], proj[:, 1], c=np.array(mean_colors)/255)
plt.legend(handles=[patches.Patch(color=np.array(colors[i])/255, label="class " + str(i)) for i in range(20)])
plt.title("feature_visualization")
plt.savefig("feature_visualization2.png")