# coding=utf-8

from skimage import io
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from data_loader_3kinds import RescaleT, RandomCrop, ToTensorLab, SalObjDataset
from model.MS-MuraNet import Net
import glob
import numpy as np
import timeit
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn

def collate_fn(batch):
    for item in batch:
        item['image'] = item['image'].float()
        item['thermal'] = item['thermal'].float()
        item['hist'] = item['hist'].float()
        item['label'] = item['label'].float()
    return {
        'image': torch.stack([b['image'] for b in batch]),
        'thermal': torch.stack([b['thermal'] for b in batch]),
        'hist': torch.stack([b['hist'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }

def save_output(image_name, pred, d_dir):
	# predict = pred
	# predict = predict.squeeze()
	# predict_np = predict.cpu().data.numpy()
	# if predict_np > 0.5 
	# im = Image.fromarray(predict_np*255).convert('RGB')
	# image = io.imread(image_name)
	# imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
	# img_name = image_name.split("/")[-1]       
	# imidx = img_name.split(".")[0]
	# imo.save(d_dir+imidx+'.png')

	predict = pred
	predict = predict.squeeze()

	predict_np = (predict.cpu().data.numpy() > 0.5).astype(np.uint8) * 255

	im = Image.fromarray(predict_np).convert('L')

	image = io.imread(image_name)
	imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

	img_name = image_name.split("/")[-1]
	imidx = img_name.split(".")[0]

	imo.save(d_dir + imidx + '.png')


def main():
	# --------- 1. get image path and name ---------

	image_dir = ''             # path of testing dataset
	prediction_dir = ''                 # path of saving results
	model_dir = ''     # path of pre-trained model
	img_name_list = glob.glob(image_dir + '*.jpg')

	# --------- 2. dataloader ---------
	test_salobj_dataset = SalObjDataset(
			img_name_list=img_name_list, 
			tra_the_dir=img_name_list,
			tra_his_dir=img_name_list,
			lbl_name_list=[],
			transform=transforms.Compose([
					RescaleT(256), 
					ToTensorLab(flag=0)
				]))
	test_salobj_dataloader = DataLoader(
			test_salobj_dataset, 
			batch_size=1, 
			shuffle=False, 
			num_workers=5, 
			collate_fn=collate_fn 
		)

	# --------- 3. model define ---------
	print("...load Net...")
	net = Net(in_channels=3)
	net.load_state_dict(torch.load(model_dir))
	net.cuda()

	net.eval()

	start = timeit.default_timer()
	# --------- 4. inference for each image ---------
	with torch.no_grad():
		for i_test, data_test in enumerate(test_salobj_dataloader):
			print("inferencing:", img_name_list[i_test].split("/")[-1])
			# inputs_test = data_test['image']
			inputs_v = data_test['image']
			thermal_image = data_test['thermal']
			hist = data_test['hist']
			inputs_v = inputs_v.type(torch.FloatTensor)
			thermal_image = thermal_image.cuda()
			hist = hist.cuda()
			inputs_v = inputs_v.cuda()

			s_out, s0, s1, s2, s3, s4, sb = net(inputs_v, thermal_image, hist)
			# s_out, s0, s1, s2, s3, s4 = net(inputs_test)

			# normalization
			pred = s_out[:, 0, :, :]
			pred = normPRED(pred)

			# save results to test_results folder
			save_output(img_name_list[i_test], pred, prediction_dir)
			del s_out, s0, s1, s2, s3, s4, sb
			# del s_out, s0, s1, s2, s3, s4

	end = timeit.default_timer()
	print("...Time...")
	print(str(end-start))


if __name__ == "__main__":
	main()

