import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils import data
from data_loader import DataLoader, ProjE_Dataset, TAProjE_Dataset
from model import ProjE, TAProjE
torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser(description='TAProjE')
parser.add_argument('--data', dest='data_dir', type=str, help="Data folder", default='./data/FB15K_D/')
parser.add_argument('--model_', dest='model_', type=str, help="Model selected", default='SSP')
parser.add_argument('--lr', dest='lr', type=float, help="Learning rate", default=0.1)
parser.add_argument("--dim", dest='dim', type=int, help="Embedding dimension", default=100)
parser.add_argument("--max_len", dest="max_len", type=int, help="Max description length", default=20)
parser.add_argument('--encoder', dest='encoder', type=str, help="Encoder type", default='nbow')
parser.add_argument('--combine_methods', dest='combine_methods', type=str, help="Text embedding and entity embedding combination methods", default='gate')
parser.add_argument('--diagonal', dest='diagonal', action="store_true", help="If dimensional attentive combination parameters use diagonal matrix or not", default=False)
parser.add_argument("--batch", dest='batch', type=int, help="Batch size", default=10)
parser.add_argument("--test_batch", dest='test_batch', type=int, help="Batch size", default=100)
parser.add_argument("--neg_weight", dest='neg_weight', type=float, help="Negative sample rate", default=0.5)
parser.add_argument("--worker", dest='n_worker', type=int, help="Evaluation worker", default=3)
parser.add_argument("--save_dir", dest='save_dir', type=str, help="Saved model path", default='./')
parser.add_argument("--resume", dest='resume', type=int, help="Resume from epoch model file", default=-1)
parser.add_argument('--pretrain', dest='pretrain', action="store_true", help="If use pretrained embedding or not", default=False)
parser.add_argument("--subset", dest='subset', type=float, help="The rate of used subset from training set, default whole training set", default=1.0)
parser.add_argument("--eval_start", dest='eval_start', type=int, help="Epoch when evaluation start", default=90)
parser.add_argument("--eval_per", dest='eval_per', type=int, help="Evaluation per x iteration", default=1)
parser.add_argument("--save_m", dest='save_m', type=int, help="Number of saved models", default=1)
parser.add_argument("--epochs", dest='epochs', type=int, help="Epochs to run", default=100)
parser.add_argument("--optimizer", dest='optimizer', type=str, help="Optimizer", default='adam')
parser.add_argument('--reg_weight', dest='reg_weight', type=float, help="The weight of L1 regularization", default=1e-5)
parser.add_argument('--dropout', dest='dropout', type=float, help="The dropout rate", default=0.5)
parser.add_argument('--clip', dest='clip', type=float, help="The gradient clip max norm", default=0.0)
parser.add_argument("--tolerance", dest="tolerance", type=int, help="Early stopping tolerance", default=3)
parser.add_argument("--early_stop_epsilon", dest="early_stop_epsilon", type=float, help="Early stopping by mean rank evaluation changes", default=0.0)

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

def use_optimizer(model, lr, weight_decay=0, lr_decay=0, momentum=0, rho=0.9, method='sgd'):
	if method=='sgd':
		return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
	elif method=='adagrad':
		return optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
	elif method=='adadelta':
		return optim.Adadelta(model.parameters(), rho=rho, lr=lr, weight_decay=weight_decay)
	elif method=='adam':
		return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	else:
		raise Exception("Invalid method, option('sgd', 'adagrad', 'adadelta', 'adam')")

def save_checkpoint(state, epoch, filename='_epoch.mod.tar'):
	remove_model = args.save_dir+str(epoch-args.save_m-1)+'_epoch.mod.tar'
	if os.path.exists(remove_model):
		 os.remove(remove_model)
	torch.save(state, filename)	

def resume(epoch):
	load_file = args.save_dir+str(epoch)+'_epoch.mod.tar'
	if os.path.isfile(load_file):	
		checkpoint = torch.load(load_file)
		print "=> loaded checkpoint '{}' (epoch {})".format(load_file, checkpoint['epoch'])	
		return checkpoint	
	else:
		print "=> no checkpoint found at '{}'".format(load_file)

def evaluation(testing_data, head_pred, tail_pred, hr_t, tr_h):
	assert len(testing_data) == len(head_pred)
	assert len(testing_data) == len(tail_pred)

	mean_rank_h = list()
	mean_rank_t = list()
	filtered_mean_rank_h = list()
	filtered_mean_rank_t = list()

	for i in range(len(testing_data)):
		h = testing_data[i, 0]
		t = testing_data[i, 1]
		r = testing_data[i, 2]
		# mean rank

		mr = 1
		for val in head_pred[i]:
			if val == h:
				mean_rank_h.append(mr)
				break
			mr += 1

		mr = 1
		for val in tail_pred[i]:
			if val == t:
				mean_rank_t.append(mr)
			mr += 1

		# filtered mean rank
		fmr = 1
		for val in head_pred[i]:
			if val == h:
				filtered_mean_rank_h.append(fmr)
				break
			if t in tr_h and r in tr_h[t] and val in tr_h[t][r]:
				continue
			else:
				fmr += 1

		fmr = 1
		for val in tail_pred[i]:
			if val == t:
				filtered_mean_rank_t.append(fmr)
				break
			if h in hr_t and r in hr_t[h] and val in hr_t[h][r]:
				continue
			else:
				fmr += 1

	return mean_rank_h, filtered_mean_rank_h, mean_rank_t, filtered_mean_rank_t

def test_ProjE(model, data_loader, test_type='test', return_rank=False, average=True): 
	acc_mean_rank_h = list()
	acc_filtered_mean_rank_h = list()
	acc_mean_rank_t = list()
	acc_filtered_mean_rank_t = list()	
	testloader = data.DataLoader(ProjE_Dataset(data_loader, dataset=test_type, neg_weight=args.neg_weight), batch_size=args.test_batch, shuffle=False, num_workers=args.n_worker)
	for batch_idx, pos_triple in enumerate(testloader):	
		triple = Variable(pos_triple)
		if use_cuda:
			triple = triple.cuda()
		head_pred, tail_pred = model.pred(triple)
		head_pred, tail_pred = head_pred.data.cpu().numpy(), tail_pred.data.cpu().numpy()
		mr_h, fmr_h, mr_t, fmr_t = evaluation(pos_triple.numpy(), head_pred, tail_pred, data_loader.hr_t, data_loader.tr_h)
		acc_mean_rank_h += mr_h
		acc_filtered_mean_rank_h += fmr_h
		acc_mean_rank_t += mr_t
		acc_filtered_mean_rank_t += fmr_t
		print 'batch %d tested' % batch_idx

	mean_rank_h = np.mean(acc_mean_rank_h)
	filtered_mean_rank_h = np.mean(acc_filtered_mean_rank_h)
	mean_rank_t = np.mean(acc_mean_rank_t)
	filtered_mean_rank_t = np.mean(acc_filtered_mean_rank_t)
	hit_10_h = np.mean(np.asarray(acc_mean_rank_h, dtype=np.int32) <= 10)
	filtered_hit_10_h = np.mean(np.asarray(acc_filtered_mean_rank_h, dtype=np.int32) <= 10)
	hit_10_t = np.mean(np.asarray(acc_mean_rank_t, dtype=np.int32) <= 10)
	filtered_hit_10_t = np.mean(np.asarray(acc_filtered_mean_rank_t, dtype=np.int32) <= 10)
	hit_1_h = np.mean(np.asarray(acc_mean_rank_h, dtype=np.int32) <= 1)
	filtered_hit_1_h = np.mean(np.asarray(acc_filtered_mean_rank_h, dtype=np.int32) <= 1)
	hit_1_t = np.mean(np.asarray(acc_mean_rank_t, dtype=np.int32) <= 1)
	filtered_hit_1_t = np.mean(np.asarray(acc_filtered_mean_rank_t, dtype=np.int32) <= 1)	

	if return_rank:
		return acc_mean_rank_h, acc_filtered_mean_rank_h, acc_mean_rank_t, acc_filtered_mean_rank_t
	else:
		if average:	
			return (mean_rank_h+mean_rank_t)/2, (filtered_mean_rank_h+filtered_mean_rank_t)/2, (
				hit_10_h+hit_10_t)/2, (filtered_hit_10_h+filtered_hit_10_t)/2, (hit_1_h+hit_1_t)/2, (filtered_hit_1_h+filtered_hit_1_t)/2	
		else:
			return mean_rank_h, filtered_mean_rank_h, mean_rank_t, filtered_mean_rank_t, hit_10_h, filtered_hit_10_h, hit_10_t, filtered_hit_10_t, hit_1_h, filtered_hit_1_h, hit_1_t, filtered_hit_1_t

def train_epoch_ProjE(model, data_loader, optimizer, epoch):
	trainloader = data.DataLoader(ProjE_Dataset(data_loader, dataset='train', neg_weight=args.neg_weight), batch_size=args.batch, shuffle=True, num_workers=args.n_worker)
	batchs = len(trainloader)
	losses = []
	for batch_idx, (triple, neg_sample_h, neg_sample_t) in enumerate(trainloader):
		triple, neg_sample_h, neg_sample_t = Variable(triple), Variable(neg_sample_h), Variable(neg_sample_t)
		if use_cuda:
			triple, neg_sample_h, neg_sample_t = triple.cuda(), neg_sample_h.cuda(), neg_sample_t.cuda()
		loss = model(triple, neg_sample_h, neg_sample_t)
		loss_ = loss.data[0]
		losses.append(loss_)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print '[Epoch %d/%d ] | Iter %d/%d | Loss %3f' % (epoch, args.epochs, batch_idx, batchs, loss_)
	return np.mean(np.array(losses))

def train_ProjE():
	data_loader = DataLoader(args.data_dir, args.subset)
	model = ProjE(args.dim, data_loader.n_entity, data_loader.n_relation, args.dropout, args.reg_weight)
	if use_cuda:
		model = model.cuda()
	optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	start_epoch = 0
	best_filtered_hit_1 = 0
	tolerance = 0
	stopped = False
	ave_losses = []
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)		
	if args.resume != -1:
		checkpoint = resume(args.resume)
		start_epoch = checkpoint['epoch']
		ave_losses = checkpoint['losses']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print model.state_dict().keys()

	for epoch in range(start_epoch, args.epochs):
		loss = train_epoch_ProjE(model, data_loader, optimizer, epoch)
		print 'Epoch %d train over, average loss: %3f' % (epoch, loss)
		ave_losses.append(loss)
		if epoch%args.eval_per==0 or epoch==args.epochs-1:		
			print 'Waiting for model validation...'
			mean_rank, filtered_mean_rank, hit_10, filtered_hit_10, hit_1, filtered_hit_1 = test_ProjE(model, data_loader, 'valid')
			print "[Validation] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f" % (
				epoch, mean_rank, filtered_mean_rank, hit_10, filtered_hit_10)	
			with open(args.save_dir+'validation_log.txt', 'ab') as f:
				f.write("[Validation] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f hit@1: %.3f filtered hit@1: %.3f loss: %.3f\n" % (
					epoch, mean_rank, filtered_mean_rank, hit_10, filtered_hit_10, hit_1, filtered_hit_1, loss))
			# Using validation set for earlystopping.
			if filtered_hit_1 - best_filtered_hit_1 <= args.early_stop_epsilon:
				tolerance += 1
				if tolerance >= args.tolerance:
					stopped = True
			else:
				tolerance = 0
				best_filtered_hit_1 = filtered_hit_1  		
			print 'Waiting for model testing...'
			mean_rank, filtered_mean_rank, hit_10, filtered_hit_10, hit_1, filtered_hit_1 = test_ProjE(model, data_loader, 'test')
			print "[Testing] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f" % (
				epoch, mean_rank, filtered_mean_rank, hit_10, filtered_hit_10)	
			with open(args.save_dir+'testing_log.txt', 'ab') as f:
				f.write("[Testing] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f hit@1: %.3f filtered hit@1: %.3f loss: %.3f\n" % (
					epoch, mean_rank, filtered_mean_rank, hit_10, filtered_hit_10, hit_1, filtered_hit_1, loss))
		save_checkpoint({
			'epoch': epoch + 1,
			'losses': ave_losses,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict()
		}, epoch, args.save_dir+str(epoch)+'_epoch.mod.tar')
		
		if stopped:
			break

def test_TAProjE(model, data_loader, test_type='test', return_rank=False, average=True): 
	acc_mean_rank_h = list()
	acc_filtered_mean_rank_h = list()
	acc_mean_rank_t = list()
	acc_filtered_mean_rank_t = list()	
	testloader = data.DataLoader(TAProjE_Dataset(data_loader, dataset=test_type, max_len=args.max_len, neg_weight=args.neg_weight), batch_size=args.test_batch, shuffle=False, num_workers=args.n_worker)
	des_list = Variable(torch.from_numpy(data_loader.get_description_list(args.max_len)), volatile=True)
	if use_cuda:
		des_list = des_list.cuda()
	des_e = model.combined_ed(des_list)
	for batch_idx, (pos_triple, hd, td) in enumerate(testloader):	
		triple, hd, td = Variable(pos_triple), Variable(hd), Variable(td)
		if use_cuda:
			triple, hd, td = triple.cuda(), hd.cuda(), td.cuda()
		head_pred, tail_pred = model.pred(triple, hd, td, des_e)
		head_pred, tail_pred = head_pred.data.cpu().numpy(), tail_pred.data.cpu().numpy()
		mr_h, fmr_h, mr_t, fmr_t = evaluation(pos_triple.numpy(), head_pred, tail_pred, data_loader.hr_t, data_loader.tr_h)
		acc_mean_rank_h += mr_h
		acc_filtered_mean_rank_h += fmr_h
		acc_mean_rank_t += mr_t
		acc_filtered_mean_rank_t += fmr_t
		print 'batch %d tested' % batch_idx

	mean_rank_h = np.mean(acc_mean_rank_h)
	filtered_mean_rank_h = np.mean(acc_filtered_mean_rank_h)
	mean_rank_t = np.mean(acc_mean_rank_t)
	filtered_mean_rank_t = np.mean(acc_filtered_mean_rank_t)
	hit_10_h = np.mean(np.asarray(acc_mean_rank_h, dtype=np.int32) <= 10)
	filtered_hit_10_h = np.mean(np.asarray(acc_filtered_mean_rank_h, dtype=np.int32) <= 10)
	hit_10_t = np.mean(np.asarray(acc_mean_rank_t, dtype=np.int32) <= 10)
	filtered_hit_10_t = np.mean(np.asarray(acc_filtered_mean_rank_t, dtype=np.int32) <= 10)
	hit_1_h = np.mean(np.asarray(acc_mean_rank_h, dtype=np.int32) <= 1)
	filtered_hit_1_h = np.mean(np.asarray(acc_filtered_mean_rank_h, dtype=np.int32) <= 1)
	hit_1_t = np.mean(np.asarray(acc_mean_rank_t, dtype=np.int32) <= 1)
	filtered_hit_1_t = np.mean(np.asarray(acc_filtered_mean_rank_t, dtype=np.int32) <= 1)		

	if return_rank:
		return acc_mean_rank_h, acc_filtered_mean_rank_h, acc_mean_rank_t, acc_filtered_mean_rank_t
	else:
		if average:	
			return (mean_rank_h+mean_rank_t)/2, (filtered_mean_rank_h+filtered_mean_rank_t)/2, (
				hit_10_h+hit_10_t)/2, (filtered_hit_10_h+filtered_hit_10_t)/2, (hit_1_h+hit_1_t)/2, (filtered_hit_1_h+filtered_hit_1_t)/2	
		else:
			return mean_rank_h, filtered_mean_rank_h, mean_rank_t, filtered_mean_rank_t, hit_10_h, filtered_hit_10_h, hit_10_t, filtered_hit_10_t, hit_1_h, filtered_hit_1_h, hit_1_t, filtered_hit_1_t

def train_epoch_TAProjE(model, data_loader, optimizer, epoch, clip):
	trainloader = data.DataLoader(TAProjE_Dataset(data_loader, dataset='train', max_len=args.max_len, neg_weight=args.neg_weight), batch_size=args.batch, shuffle=True, num_workers=args.n_worker)
	des_list = Variable(torch.from_numpy(data_loader.get_description_list(args.max_len)))
	if use_cuda:
		des_list = des_list.cuda()
	batchs = len(trainloader)
	losses = []
	for batch_idx, (triple, hd, td, neg_sample_h, neg_sample_t) in enumerate(trainloader):
		triple, hd, td, neg_sample_h, neg_sample_t = Variable(triple), Variable(hd), Variable(td), Variable(
			neg_sample_h), Variable(neg_sample_t)
		if use_cuda:
			triple, hd, td, neg_sample_h, neg_sample_t = triple.cuda(), hd.cuda(), td.cuda(
				), neg_sample_h.cuda(), neg_sample_t.cuda()
		loss = model(triple, hd, td, neg_sample_h, neg_sample_t, des_list)
		loss_ = loss.data[0]
		if loss_!=loss_:
			return loss_
		losses.append(loss_)
		optimizer.zero_grad()
		loss.backward()
		if clip!=0:
			nn.utils.clip_grad_norm(model.parameters(), clip)
		optimizer.step()
		print '[Epoch %d/%d ] | Iter %d/%d | Loss %3f' % (epoch, args.epochs, batch_idx, batchs, loss_)
	return np.mean(np.array(losses))

def train_TAProjE():
	data_loader = DataLoader(args.data_dir, args.subset)
	model = TAProjE(args.dim, data_loader.n_entity, data_loader.n_relation, data_loader.n_word, args.dropout, args.reg_weight, args.encoder, args.combine_methods, args.diagonal, args.pretrain)
	if use_cuda:
		model = model.cuda()
	optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	start_epoch = 0
	best_filtered_hit_10 = 0
	tolerance = 0
	stopped = False
	ave_losses = []
	clip = args.clip
	# lowest_loss = 10000
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)		
	if args.resume != -1:
		checkpoint = resume(args.resume)
		start_epoch = checkpoint['epoch']
		clip = checkpoint['clip']
		ave_losses = checkpoint['losses']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print model.state_dict().keys()

	for epoch in range(start_epoch, args.epochs):
		loss = train_epoch_TAProjE(model, data_loader, optimizer, epoch, clip)
		while loss!=loss:
			if clip==0:
				clip=1.0
			else:
				clip=clip/10
			if epoch!=0:
				res = resume(epoch-1)
				model.load_state_dict(res['state_dict'])
				optimizer.load_state_dict(res['optimizer'])
			else:
				print 'reinitialize model......'
				model = TAProjE(args.dim, data_loader.n_entity, data_loader.n_relation, data_loader.n_word, args.dropout, args.reg_weight, args.encoder, args.combine_methods, args.diagonal, args.pretrain)
				if use_cuda:
					model = model.cuda()
				optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)				
			loss = train_epoch_TAProjE(model, data_loader, optimizer, epoch, clip)
		print 'Epoch %d train over, average loss: %3f' % (epoch, loss)
		ave_losses.append(loss)
		if epoch%args.eval_per==0 or epoch==args.epochs-1:
			print 'Waiting for model validation...'
			mean_rank, filtered_mean_rank, hit_10, filtered_hit_10, hit_1, filtered_hit_1 = test_TAProjE(model, data_loader, 'valid')
			print "[Validation] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f" % (
				epoch, mean_rank, filtered_mean_rank, hit_10, filtered_hit_10)	
			with open(args.save_dir+'validation_log.txt', 'ab') as f:
				f.write("[Validation] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f hit@1: %.3f filtered hit@1: %.3f loss: %.3f\n" % (
					epoch, mean_rank, filtered_mean_rank, hit_10, filtered_hit_10, hit_1, filtered_hit_1, loss))
			# Using validation set for earlystopping.
			if filtered_hit_10 - best_filtered_hit_10 <= args.early_stop_epsilon:
				tolerance += 1
				if tolerance >= args.tolerance:
					stopped = True
			else:
				tolerance = 0
				best_filtered_hit_10 = filtered_hit_10  		
			print 'Waiting for model testing...'
			mean_rank, filtered_mean_rank, hit_10, filtered_hit_10, hit_1, filtered_hit_1 = test_TAProjE(model, data_loader, 'test')
			print "[Testing] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f" % (
				epoch, mean_rank, filtered_mean_rank, hit_10, filtered_hit_10)	
			with open(args.save_dir+'testing_log.txt', 'ab') as f:
				f.write("[Testing] epoch %d mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f hit@1: %.3f filtered hit@1: %.3f loss: %.3f\n" % (
					epoch, mean_rank, filtered_mean_rank, hit_10, filtered_hit_10, hit_1, filtered_hit_1, loss))
		save_checkpoint({
			'epoch': epoch + 1,
			'clip': clip,
			'losses': ave_losses,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict()
		}, epoch, args.save_dir+str(epoch)+'_epoch.mod.tar')
		if stopped:
			break

def get_rank():
	out_dir = 'results/FB15K0.1/'
	data_loader = DataLoader(args.data_dir)	
	des_list = Variable(torch.from_numpy(data_loader.get_description_list(args.max_len)), volatile=True)
	if use_cuda:
		des_list = des_list.cuda()	
	if args.model_=='ProjE':
		model = ProjE(args.dim, data_loader.n_entity, data_loader.n_relation, args.dropout, args.reg_weight)
	elif args.model_=='TAProjE':			
		model = TAProjE(args.dim, data_loader.n_entity, data_loader.n_relation, data_loader.n_word, args.dropout, args.reg_weight, args.encoder, args.combine_methods, args.diagonal, args.pretrain)
	if use_cuda:
		model = model.cuda()	
	optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)		
	if args.resume != -1:
		checkpoint = resume(args.resume)
		start_epoch = checkpoint['epoch']
		ave_losses = checkpoint['losses']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print model.state_dict().keys()	
		if args.model_=='ProjE':
			acc_mean_rank_h, acc_filtered_mean_rank_h, acc_mean_rank_t, acc_filtered_mean_rank_t = test_ProjE(model, data_loader, 'test', return_rank=True)
			fmr = (np.asarray(acc_filtered_mean_rank_h, dtype=np.float32)+np.asarray(acc_filtered_mean_rank_t, dtype=np.float32))/2
			with open(out_dir+'ProjE_MR_log.txt', 'ab') as f:
				for i in range(len(fmr)):
					f.write(str(fmr[i])+'\n')
		elif args.model_=='TAProjE':	
			acc_mean_rank_h, acc_filtered_mean_rank_h, acc_mean_rank_t, acc_filtered_mean_rank_t = test_TAProjE(model, data_loader, 'test', return_rank=True)	
			fmr = (np.asarray(acc_filtered_mean_rank_h, dtype=np.float32)+np.asarray(acc_filtered_mean_rank_t, dtype=np.float32))/2			
			if args.combine_methods=='gate':
				with open(out_dir+'TAProjE_GATE_MR_log.txt', 'ab') as f:
					for i in range(len(fmr)):
						f.write(str(fmr[i])+'\n')				
			elif args.combine_methods=='dimensional_attentive':
				if args.diagonal:
					with open(out_dir+'TAProjE_DAC_diag_MR_log.txt', 'ab') as f:
						for i in range(len(fmr)):
							f.write(str(fmr[i])+'\n')
				else:
					with open(out_dir+'TAProjE_DAC_MR_log.txt', 'ab') as f:
						for i in range(len(fmr)):
							f.write(str(fmr[i])+'\n')							

def relation_cat():
	out_dir = 'analysis/'
	data_loader = DataLoader(args.data_dir)	
	des_list = Variable(torch.from_numpy(data_loader.get_description_list(args.max_len)), volatile=True)
	if use_cuda:
		des_list = des_list.cuda()	
	if args.model_=='ProjE':
		model = ProjE(args.dim, data_loader.n_entity, data_loader.n_relation, args.dropout, args.reg_weight)
	elif args.model_=='TAProjE':			
		model = TAProjE(args.dim, data_loader.n_entity, data_loader.n_relation, data_loader.n_word, args.dropout, args.reg_weight, args.encoder, args.combine_methods, args.diagonal, args.pretrain)
	if use_cuda:
		model = model.cuda()	
	optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)		
	if args.resume != -1:
		checkpoint = resume(args.resume)
		start_epoch = checkpoint['epoch']
		ave_losses = checkpoint['losses']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print model.state_dict().keys()
	evaluation_list = ['one2one', 'one2many', 'many2one', 'many2many']	
	for r_type in evaluation_list:
		if args.model_=='ProjE':
			mean_rank_h, filtered_mean_rank_h, mean_rank_t, filtered_mean_rank_t, hit_10_h, filtered_hit_10_h, hit_10_t, filtered_hit_10_t, hit_1_h, filtered_hit_1_h, hit_1_t, filtered_hit_1_t = test_ProjE(model, data_loader, r_type, average=False)
			with open(out_dir+'ProjE_RC_pred_h_log.txt', 'ab') as f:
				f.write("[%s] mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f hit@1: %.3f filtered hit@1: %.3f\n" % (
					r_type, mean_rank_h, filtered_mean_rank_h, hit_10_h, filtered_hit_10_h, hit_1_h, filtered_hit_1_h))	
			with open(out_dir+'ProjE_RC_pred_t_log.txt', 'ab') as f:
				f.write("[%s] mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f hit@1: %.3f filtered hit@1: %.3f\n" % (
					r_type, mean_rank_t, filtered_mean_rank_t, hit_10_t, filtered_hit_10_t, hit_1_t, filtered_hit_1_t))	
		elif args.model_=='TAProjE':	
			mean_rank_h, filtered_mean_rank_h, mean_rank_t, filtered_mean_rank_t, hit_10_h, filtered_hit_10_h, hit_10_t, filtered_hit_10_t, hit_1_h, filtered_hit_1_h, hit_1_t, filtered_hit_1_t = test_TAProjE(model, data_loader, r_type, average=False)	
			if args.combine_methods=='gate':
				with open(out_dir+'TAProjE_GATE_RC_pred_h_log.txt', 'ab') as f:
					f.write("[%s] mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f hit@1: %.3f filtered hit@1: %.3f\n" % (
						r_type, mean_rank_h, filtered_mean_rank_h, hit_10_h, filtered_hit_10_h, hit_1_h, filtered_hit_1_h))
				with open(out_dir+'TAProjE_GATE_RC_pred_t_log.txt', 'ab') as f:
					f.write("[%s] mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f hit@1: %.3f filtered hit@1: %.3f\n" % (
						r_type, mean_rank_t, filtered_mean_rank_t, hit_10_t, filtered_hit_10_t, hit_1_t, filtered_hit_1_t))					
			elif args.combine_methods=='dimensional_attentive':
				if args.diagonal:
					with open(out_dir+'TAProjE_DAC_diagonal_RC_pred_h_log.txt', 'ab') as f:
						f.write("[%s] mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f hit@1: %.3f filtered hit@1: %.3f\n" % (
							r_type, mean_rank_h, filtered_mean_rank_h, hit_10_h, filtered_hit_10_h, hit_1_h, filtered_hit_1_h))
					with open(out_dir+'TAProjE_DAC_diagonal_RC_pred_t_log.txt', 'ab') as f:
						f.write("[%s] mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f hit@1: %.3f filtered hit@1: %.3f\n" % (
							r_type, mean_rank_t, filtered_mean_rank_t, hit_10_t, filtered_hit_10_t, hit_1_t, filtered_hit_1_t))	
				else:			
					with open(out_dir+'TAProjE_DAC_RC_pred_h_log.txt', 'ab') as f:
						f.write("[%s] mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f hit@1: %.3f filtered hit@1: %.3f\n" % (
							r_type, mean_rank_h, filtered_mean_rank_h, hit_10_h, filtered_hit_10_h, hit_1_h, filtered_hit_1_h))
					with open(out_dir+'TAProjE_DAC_RC_pred_t_log.txt', 'ab') as f:
						f.write("[%s] mean rank: %.3f filtered mean rank: %.3f hit@10: %.3f filtered hit@10: %.3f hit@1: %.3f filtered hit@1: %.3f\n" % (
							r_type, mean_rank_t, filtered_mean_rank_t, hit_10_t, filtered_hit_10_t, hit_1_t, filtered_hit_1_t))						
									
def get_des_weight():
	out_dir = 'results/FB15K0.1/'
	data_loader = DataLoader(args.data_dir)	
	des_list = Variable(torch.from_numpy(data_loader.get_description_list(args.max_len)), volatile=True)
	if use_cuda:
		des_list = des_list.cuda()	
	model = TAProjE(args.dim, data_loader.n_entity, data_loader.n_relation, data_loader.n_word, args.dropout, args.reg_weight, args.encoder, args.combine_methods, args.diagonal, args.pretrain)
	if use_cuda:
		model = model.cuda()	
	optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)		
	if args.resume != -1:
		checkpoint = resume(args.resume)
		start_epoch = checkpoint['epoch']
		ave_losses = checkpoint['losses']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print model.state_dict().keys()
	w = model.des_weght(des_list).data.cpu().numpy()
	des_w = np.mean(w, axis=1)
	# des_w = w[:, 0].reshape((-1))
	if args.combine_methods=='gate':
		des_weight = open(out_dir+'des_gate.txt', 'ab')
	elif args.combine_methods=='dimensional_attentive':
		if args.diagonal:
			des_weight = open(out_dir+'des_DAC_diag.txt', 'ab')	
		else:			
			des_weight = open(out_dir+'des_DAC.txt', 'ab')		
	for i in range(len(des_w)):
		des_weight.write(str(des_w[i])+'\n')
	des_weight.close()

def dimensional_rank():
	out_dir = 'analysis/WN18/'
	data_loader = DataLoader(args.data_dir)	
	des_list = Variable(torch.from_numpy(data_loader.get_description_list(args.max_len)), volatile=True)
	if use_cuda:
		des_list = des_list.cuda()	
	model = TAProjE(args.dim, data_loader.n_entity, data_loader.n_relation, data_loader.n_word, args.dropout, args.reg_weight, args.encoder, args.combine_methods, args.diagonal, args.pretrain)
	if use_cuda:
		model = model.cuda()	
	optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)		
	if args.resume != -1:
		checkpoint = resume(args.resume)
		start_epoch = checkpoint['epoch']
		ave_losses = checkpoint['losses']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print model.state_dict().keys()
	w = model.des_weght(des_list).data.cpu().numpy().T
	top_n = np.argsort(w, 1)
	e_top_10 = top_n[:, :10]
	d_top_10 = top_n[:,::-1][:, :10]
	etop10 = open(out_dir+'e_top_10.txt', 'ab')
	dtop10 = open(out_dir+'d_top_10.txt', 'ab')
	shape_ = e_top_10.shape
	for i in range(shape_[0]):
		s_e = ''
		s_d = ''
		for j in range(shape_[1]):
			s_e += str(e_top_10[i][j])+' '
			s_d += str(d_top_10[i][j])+' '
		etop10.write(s_e[:-1]+'\n')
		dtop10.write(s_d[:-1]+'\n')
	etop10.close()
	dtop10.close()

def get_model_param():
	data_loader = DataLoader(args.data_dir)
	model = ProjE(args.dim, data_loader.n_entity, data_loader.n_relation, args.dropout, args.reg_weight)
	if use_cuda:
		model = model.cuda()
	optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)		
	if args.resume != -1:
		checkpoint = resume(args.resume)
		start_epoch = checkpoint['epoch']
		ave_losses = checkpoint['losses']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print model.state_dict().keys()

	np.save('pretrain_embedding/entity_embedding', model.entity_embedding.weight.data.cpu().numpy())
	np.save('pretrain_embedding/relation_embedding', model.relation_embedding.weight.data.cpu().numpy())
	np.save('pretrain_embedding/weight_h', model.weight_h.data.cpu().numpy())
	np.save('pretrain_embedding/weight_hr', model.weight_hr.data.cpu().numpy())
	np.save('pretrain_embedding/weight_t', model.weight_t.data.cpu().numpy())
	np.save('pretrain_embedding/weight_tr', model.weight_tr.data.cpu().numpy())	
	np.save('pretrain_embedding/weight_bias_h', model.weight_bias_h.data.cpu().numpy())
	np.save('pretrain_embedding/weight_bias_t', model.weight_bias_t.data.cpu().numpy())
	print 'OK'
	# data_loader = DataLoader(args.data_dir)
	# model = ProjE_D(args.dim, data_loader.n_entity, data_loader.n_relation, data_loader.n_word, args.dropout, args.reg_weight)
	# if use_cuda:
	# 	model = model.cuda()
	# optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	# # lowest_loss = 10000
	# if not os.path.exists(args.save_dir):
	# 	os.makedirs(args.save_dir)		
	# if args.resume != '':
	# 	if os.path.isfile(args.resume):
	# 		print "=> loading checkpoint '{}'".format(args.resume)
	# 		checkpoint = torch.load(args.resume)
	# 		start_epoch = checkpoint['epoch']
	# 		ave_losses = checkpoint['losses']
	# 		model.load_state_dict(checkpoint['state_dict'])
	# 		optimizer.load_state_dict(checkpoint['optimizer'])
	# 		print model.state_dict().keys()
	# 		print "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch'])
	# 	else:
	# 		print "=> no checkpoint found at '{}'".format(args.resume)	
	# print model.gate.weight.data.cpu().numpy()[2]


def main():
	print args
	# get_model_param()
	if args.model_=='ProjE':
		train_ProjE()
	elif args.model_=='TAProjE':
		train_TAProjE()

if __name__ == '__main__':
	main()
	# get_des_weight()
	# dimensional_rank()
	# relation_cat()
	# get_rank()


