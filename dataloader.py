from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from os.path import join
from PIL import Image

class CelebDataSet(Dataset):
    """CelebA dataset
    Parameters:
        data_path (str)     -- CelebA dataset main directory(inculduing '/Img' and '/Anno') path
        state (str)         -- dataset phase 'train' | 'val' | 'test'

    Center crop the alingned celeb dataset to 178x178 to include the face area and then downsample to 128x128(Step3).
    In addition, for progressive training, the target image for each step is resized to 32x32(Step1) and 64x64(Step2).
    """

    def __init__(self, data_path = './dataset/', state = 'train', data_augmentation=None):
        self.main_path = data_path
        self.state = state
        self.data_augmentation = data_augmentation

        self.img_path = join(self.main_path, 'CelebA/Img/img_align_celeba')
        self.eval_partition_path = join(self.main_path, 'Anno/list_eval_partition.txt')

        train_img_list = []
        val_img_list = []
        test_img_list = []

        f = open(self.eval_partition_path, mode='r')

        while True:
            line = f.readline().split()

            if not line: break

            if line[1] == '0':
                train_img_list.append(line)
            elif line[1] =='1':
                val_img_list.append(line)
            else:
                test_img_list.append(line)

        f.close()

        if state=='train':
            train_img_list.sort()
            self.image_list = train_img_list
        elif state=='val':
            val_img_list.sort()
            self.image_list = val_img_list
        else:
            test_img_list.sort()
            self.image_list = test_img_list

        if state=='train' and self.data_augmentation:
            self.pre_process = transforms.Compose([
                                                transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop((178, 178)),
                                                transforms.Resize((128, 128)),
                                                transforms.RandomRotation(20, resample=Image.BILINEAR),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                                               ])
        else:
            self.pre_process = transforms.Compose([
                                            transforms.CenterCrop((178, 178)),
                                            transforms.Resize((128,128)),
                                            ])

        self.totensor = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

        self._64x64_down_sampling = transforms.Resize((64, 64))
        self._32x32_down_sampling = transforms.Resize((32, 32))
        self._16x16_down_sampling = transforms.Resize((16,16))

    def __getitem__(self, index):
        image_path = join(self.img_path, self.image_list[index][0])
        target_image = Image.open(image_path).convert('RGB')
        target_image = self.pre_process(target_image)
        x4_target_image = self._64x64_down_sampling(target_image)
        x2_target_image = self._32x32_down_sampling(x4_target_image)
        input_image = self._16x16_down_sampling(x2_target_image)

        x2_target_image = self.totensor(x2_target_image)
        x4_target_image = self.totensor(x4_target_image)
        target_image = self.totensor(target_image)
        input_image = self.totensor(input_image)

        return x2_target_image, x4_target_image, target_image, input_image

    def __len__(self):
        return len(self.image_list)
