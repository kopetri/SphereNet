from pathlib import Path
import torch
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF

class Structured3DDataset(Dataset):
    def __init__(self, path, split, shuffle=True):
        super().__init__()
        self.split = split
        self.path = Path(path)
        self.scenes = self.load_scenes()
        self.imgs = self.load_images()
        self.depths = self.load_depth()
        self.shuffle = shuffle
        print("Found {} images for split {}.".format(len(self.imgs), self.split))

    def get_XYZ(self, u, v, depth):
        H, W = depth.shape
        vert_fov = np.pi
        vertical_angle = vert_fov / 2 - (v * vert_fov) / H
        depth_value = depth[v,u] 
        horizontal_angle = (u * 2 * np.pi) / W
        return self.get_location_from_angles(vertical_angle, horizontal_angle, depth_value)

    def get_location_from_angles(self, vertical_angle, horizontal_angle, depth_value=1.0):
        X = - np.cos(vertical_angle) * (np.sin(horizontal_angle) * depth_value)
        Y = - np.cos(vertical_angle) * (np.cos(horizontal_angle) * depth_value) 
        Z = (np.sin(vertical_angle) * depth_value)
        return [X, Y, Z]

    def load_scenes(self):
        if self.split == "train":
            return [s.name for s in self.path.glob("*") if s.is_dir() and 0 <= int(s.name.replace("scene_", "")) <= 2999] # scene_00000 to scene_02999
        elif self.split == "valid":
            return [s.name for s in self.path.glob("*") if s.is_dir() and 3000 <= int(s.name.replace("scene_", "")) <= 3249] # scene_03000 to scene_03249
        elif self.split == "test":
            return [s.name for s in self.path.glob("*") if s.is_dir() and 3250 <= int(s.name.replace("scene_", "")) <= 3499] # scene_03250 to scene_03499

    def load_images(self):
        return [img for img in self.path.glob("**/*") if img.parent.name == "full" and img.stem == "rgb_rawlight" and img.parents[4].name in self.scenes]

    def load_depth(self):
        return [depth for depth in self.path.glob("**/*") if depth.parent.name == "empty" and depth.stem == "depth" and depth.parents[4].name in self.scenes]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        depth = self.depths[idx]

        assert img.parents[2].name == depth.parents[2].name

        img = cv2.imread(img.as_posix(), cv2.IMREAD_COLOR)
        H,W,C = img.shape
        depth = cv2.imread(depth.as_posix(), cv2.IMREAD_ANYDEPTH)

        U = np.tile(np.arange(0, W), [H, 1])
        V = np.repeat(np.arange(0, H), W).reshape((H, W))

        U = U.reshape((-1))
        V = V.reshape((-1))
        img = img.reshape((-1, 3))
        depth = depth.reshape((-1))
 

        indices = np.arange(0, H * W)
        if self.shuffle:
            np.random.shuffle(indices)
        indices = indices[0:1088]
        

        U = U[indices]
        V = V[indices]
        color = img[indices]
        depth = depth[indices]


        vert_fov = np.pi   
        vertical_angle = vert_fov / 2 - (V * vert_fov) / H
        horizontal_angle = (U * 2 * np.pi) / W


        xyz = np.zeros((3, 1088), dtype=np.float32)
        xyz[0, :] = - np.cos(vertical_angle) * (np.sin(horizontal_angle))
        xyz[1, :] = - np.cos(vertical_angle) * (np.cos(horizontal_angle)) 
        xyz[2, :] =   np.sin(vertical_angle)

        target = xyz * depth
        assert not np.any(np.isnan(target)), "target has NaN"

        xyz = np.transpose(xyz, (1, 0))
        target = np.transpose(target, (1, 0))


        minimum = np.min(target, axis=0)
        maximum = np.max(target, axis=0)

        assert all([v > 0 for v in (maximum - minimum)]), "wrong xyz: {}".format(maximum - minimum)

        
        center = minimum + 0.5 * (maximum - minimum)
        
        target -= center # center around 0
        assert not np.any(np.isnan(target)), "target has NaN"

        minimum = np.min(target, axis=0)
        maximum = np.max(target, axis=0)

        dim = maximum - minimum
        assert all([v > 0 for v in dim]), "dim == 0: {}".format(dim)
        
        target /= dim # normalize to -0.5 and 0.5
        assert not np.any(np.isnan(target)), "target has NaN"
        target *= 2.0 #     scale to -1.0 and 1


        color = torch.from_numpy(color) / 255.0
        xyz = torch.from_numpy(xyz)
        target = torch.from_numpy(target)

        X = torch.cat([xyz, color], dim=1)

        return X.permute(1,0), target.permute(1,0)

        

        

       



if __name__ == '__main__':
    dataset = Structured3DDataset("F:/data/Structured3D", "train", shuffle=False)
    i = 0
    for d in dataset:
        if i > 100:
            break
        i += 1