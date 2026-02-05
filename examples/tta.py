from __future__ import print_function, absolute_import

import argparse
import os
import os.path as osp
import pickle
import random
import shutil
from gcl.datasets.query import collect_gan_images
import numpy as np

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision
from gcl import datasets
from gcl import models
from gcl.trainer import DGNet_Trainer
from gcl.utils.data import transforms as T
from gcl.utils.data.preprocessor import Preprocessor, AllMeshPreprocessor
from gcl.utils.serialization import load_checkpoint
from gcl.utils.gan_utils import get_config
#from gcl.evaluators import Evaluator
from gcl.evaluators import evaluate_all
from torchvision import utils
from tqdm import tqdm
from gcl.evaluators import extract_features
from gcl.ranker import Ranker
from gcl.utils.features import select_topk_expansions

start_epoch = best_mAP = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#method = 'min' # 'avg', 'center'
method = 'center'
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def get_display_loader(dataset, height, width, batch_size, workers, testset=None, mesh_dir=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    mesh_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    if (testset is None):
        testset = dataset.gallery
        mesh_dir = mesh_dir + 'test/'
    else:
        mesh_dir = mesh_dir + 'train/'

    test_loader = DataLoader(
        AllMeshPreprocessor(testset, root=dataset.images_dir, transform=test_transformer, mesh_dir=mesh_dir,
                            mesh_transform=mesh_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader



def get_GAN_loader(list, height, width, batch_size, workers):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    gan_loader = DataLoader(
        Preprocessor(list,root="",transform=transformer),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True
    )
    return gan_loader

def create_model(args):
    model_1 = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=0)

    if args.init == '':
        print('No idnet init.')
    else:
        checkpoint = load_checkpoint(args.init)
        model_1.load_state_dict(checkpoint, strict=False)

    model_1.cuda()

    return model_1


def denormalize_recon(x):
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    x_recon = (x * std) + mean

    return x_recon

def denormalize_mesh(x):
    mean = torch.FloatTensor([0.5]).cuda()
    std = torch.FloatTensor([0.5]).cuda()
    x_recon = (x * std) + mean

    return x_recon
'''
def generate_nv(trainer, train_display_loader, path, degree):
    for i, (img, mesh_org, all_mesh_nv, fname, pid, camid, index) in enumerate(tqdm(train_display_loader)):
        idx = int(degree / 45 - 1)
        img_name = fname[0].split('/')[-1]  # image name

        img = img.cuda()
        mesh_nv = all_mesh_nv[idx].cuda()

        img_nv = trainer.sample_nv(img, mesh_nv)
        img_nv = denormalize_recon(img_nv)

        utils.save_image(
            img_nv,
            osp.join(path, img_name),
            normalize=True,
            range=(0, 1.0),
        )
    '''

def generate_nv(trainer, mesh_loader, query, output_path):
    mesh_iter = iter(mesh_loader)

    #Prova a passare la query in gray e quando devi salvare direttamente la foto prima fai il decode
    for i in [45, 90, 135, 180, 225, 270, 315]:
        save_path = osp.join(output_path, "nv", str(i))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    for i,(img, fname, pid, camid, index) in enumerate(tqdm(query)):
        '''# Prendo l'indice di una persona casualmente
        rand_idx = random.randint(0, len(mesh_loader) - 1)
        # Recupero la mesh originale e le mesh ruotate
        _, _, all_mesh_nv, _, _, _, _ = mesh_loader.dataset[rand_idx]
        '''
        try:
            _, mesh_org, all_mesh_nv, _, _, _, _ = next(mesh_iter)
        except StopIteration:
            mesh_iter = iter(mesh_loader)
            _, mesh_org, all_mesh_nv, _, _, _, _ = next(mesh_iter)

        
        #recupero la cartella dove sono contenuti i dati (potrei passarla direttamente alla funzione) 
        orig_path = fname[0]
        dest_path = os.path.join(output_path, "nv", "0", os.path.basename(orig_path))

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        #shutil.copy(orig_path, dest_path)

        # copia dell'immagine originale
        #shutil.copy(orig_path, dest_path)
        img = img.cuda()
        mesh_org = mesh_org.cuda()

        img_nv = trainer.sample_nv(img, mesh_org)
        img_nv = denormalize_recon(img_nv)

        utils.save_image(
            img_nv,
            dest_path,
            normalize=True,
            range=(0, 1.0),
        )

        for j in [45, 90, 135, 180, 225, 270, 315]:
            idx = int(j / 45 - 1)
            img_name = fname[0].split('/')[-1]  # image name

            img = img.cuda()
            mesh_nv = all_mesh_nv[idx].cuda()

            '''if mesh_nv.dim() == 3:        # [C, H, W]
                mesh_nv = mesh_nv.unsqueeze(0)
            elif mesh_nv.dim() == 2:      # [H, W]
                mesh_nv = mesh_nv.unsqueeze(0).unsqueeze(0)
            '''

            #print(img.shape)
            #print(mesh_nv)

            img_nv = trainer.sample_nv(img, mesh_nv)
            img_nv = denormalize_recon(img_nv)

            path = osp.join(output_path, "nv", str(j))
            utils.save_image(
                img_nv,
                osp.join(path, img_name),
                normalize=True,
                range=(0, 1.0),
            )

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True

    config = get_config(args.config)
    print("==========\nArgs:{}\n==========".format(args))


    # loading model
    print('==> loading model')
    if args.dataset_target == 'market1501':
        #checkpoint_path = 'outputs/market_init_JVTC_unsupervised/checkpoints'
        checkpoint_path = 'outputs/duke_init_JVTC_unsupervised/checkpoints'

        model_1 = create_model(args)
        trainer = DGNet_Trainer(config, model_1, args.idnet_fix).cuda()
        iterations = trainer.resume(checkpoint_path, hyperparameters=config)
        #iterations = 0
        output_path = 'outputs/market_init_JVTC_unsupervised/images'
        os.makedirs(output_path, exist_ok=True)
        feat_path = 'outputs/market_init_JVTC_unsupervised/feat'
    elif args.dataset_target == 'dukemtmc-reid':
        checkpoint_path = 'outputs/duke_init_JVTC_unsupervised/checkpoints'
        model_1 = create_model(args)
        trainer = DGNet_Trainer(config, model_1, args.idnet_fix).cuda()
        iterations = trainer.resume(checkpoint_path, hyperparameters=config)
        #iterations = 0
        output_path = 'outputs/duke_init_JVTC_unsupervised/images'
        os.makedirs(output_path, exist_ok=True)
        feat_path = 'outputs/duke_init_JVTC_unsupervised/feat'
    elif args.dataset_target == 'msmt17':
        checkpoint_path = 'outputs/msmt_init_JVTC_unsupervised/checkpoints'
        model_1 = create_model(args)
        trainer = DGNet_Trainer(config, model_1, args.idnet_fix).cuda()
        iterations = trainer.resume(checkpoint_path, hyperparameters=config)
        #iterations = 0
        output_path = 'outputs/msmt_init_JVTC_unsupervised/images'
        os.makedirs(output_path, exist_ok=True)
        feat_path = 'outputs/msmt_init_JVTC_unsupervised/feat'
    else:
        raise NotImplementedError

    output_path_query =  osp.join(output_path, "query")
    output_path_gallery = osp.join(output_path, "gallery")

    # prepare dataset
    print('==> preparing dataset')
    # Create data loaders
    dataset_target = get_data(args.dataset_target, args.data_dir)


    #query_loader = get_query_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    query_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers,testset=dataset_target.query)

    #gallery_loader = get_display_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, mesh_dir=args.mesh_dir)
    gallery_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers,testset=dataset_target.gallery)
     
    # # evaluation
    # test_loader_target = get_test_loader(dataset_target, args.height, args.width, 128, args.workers)
    # evaluator_1 = Evaluator(model_1)
    # _, mAP_1 = evaluator_1.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    # display set
    #train_display_loader = get_display_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, testset=dataset_target.train, mesh_dir=args.mesh_dir)
    #train_display_loader = get_query_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, mesh_dir=args.mesh_dir)
    train_display_loader = get_display_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, mesh_dir=args.mesh_dir)

    # generate data
    if args.gen == 1: 
        '''
        save_path = osp.join(output_path, args.mode, str(args.degree))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        generate_nv(trainer, train_display_loader, save_path, degree=args.degree)
        '''
        generate_nv(trainer,train_display_loader,query_loader,output_path_query)
        generate_nv(trainer,train_display_loader,gallery_loader,output_path_gallery)
        print('\n=============IMMAGINI GENERATE CORRETTAMENTE=============\n')
    elif args.gen == 0:
        print('\n===============NESSUNA IMMAGINE GENERATA=================\n')
        
        #(galleryCams, galleryDescriptors, galleryIds, galleryPaths, queryCams, queryDescriptors, queryIds, queryPaths) 
        os.makedirs(feat_path, exist_ok=True)
        #local feature
        lf = args.lf

        if lf:
            try:
                with open(os.path.join(feat_path, "queryFeatures.pkl"), "rb") as f:
                    queryFeatures = pickle.load(f)
                print("Query features loaded successfully\n")
            except Exception as e:
                print(f"Error loading query features: {e}")
                raise
            try:
                with open(os.path.join(feat_path, "galleryFeatures.pkl"), "rb") as f:
                    galleryFeatures = pickle.load(f)
                print("Gallery features loaded successfully\n")
            except Exception as e:
                print(f"Error loading query features: {e}")
                raise
        else:
            print("[QUERY EXTRACTION]\n")
            queryFeatures, _ = extract_features(trainer.id_net, query_loader)
            with open(os.path.join(feat_path, "queryFeatures.pkl"), "wb") as f:
                pickle.dump(queryFeatures, f)

            print("\n[GALLERY EXTRACTION]\n")
            galleryFeatures, _ = extract_features(trainer.id_net, gallery_loader)
            with open(os.path.join(feat_path, "galleryFeatures.pkl"), "wb") as f:
                pickle.dump(galleryFeatures, f)

        query_list = dataset_target.query

        queryPaths = [x[0] for x in query_list]
        queryIds   = [x[1] for x in query_list]
        queryCams  = [x[2] for x in query_list]
        queryDescriptors = torch.stack([queryFeatures[p] for p in queryPaths], dim=0)


        gallery_list = dataset_target.gallery
        galleryPaths = [x[0] for x in gallery_list]
        galleryIds   = [x[1] for x in gallery_list]
        galleryCams  = [x[2] for x in gallery_list]
        galleryDescriptors = torch.stack([galleryFeatures[p] for p in galleryPaths], dim=0)

        # ---- CONTROLLI DIMENSIONI ----
        print("Gallery:")
        print("  paths:", len(galleryPaths))
        print("  ids:  ", len(galleryIds))
        print("  cams: ", len(galleryCams))
        print("  descriptors:", galleryDescriptors.shape)
        #print(galleryPaths[:3])

        print("\nQuery:")
        print("  paths:", len(queryPaths))
        print("  ids:  ", len(queryIds))
        print("  cams: ", len(queryCams))
        print("  descriptors:", queryDescriptors.shape)

        N = len(queryIds) # number of query images
        M = len(galleryIds) # number of gallery images
        #(galleryGANCams, galleryGANDescriptors, galleryGANIds, galleryGANPaths, queryGANCams, queryGANDescriptors, queryGANIds, queryGANPaths) 
        
        #output_path = "outputs/market_init_JVTC_unsupervised/images/queryaugmentation" #Ricordati di cambiarlo
        #save_path_query = osp.join(output_path,"queryaugmentation", "nv")
        save_path_query = osp.join(output_path_query, "nv")
        save_path_gallery = osp.join(output_path_gallery, "nv")

        #In tutte le cartelle dovrebbero esserci gli stessi file

        print("\n[COLLECTING QUERY GAN INFORMATION]\n")
        queryGANPaths, queryGANIds, queryGANCams, gan_query_list = collect_gan_images(dataset_target.query,gan_root=save_path_query)
        print("\n[COLLECTING GALLERY GAN INFORMATION]\n")
        galleryGANPaths, galleryGANIds, galleryGANCams, gan_gallery_list = collect_gan_images(dataset_target.gallery,gan_root=save_path_gallery)
       
        if lf:
            print("\n[LOADING SAVED FEATURES]\n")
            
            try:
                with open(os.path.join(feat_path, "queryGANDescriptors.pkl"), "rb") as f:
                    queryGANDescriptors = pickle.load(f)
                print("Query GAN features loaded successfully\n")
            except Exception as e:
                print(f"Error loading query GAN features: {e}")
                raise

            # Load GALLERY features
            try:
                with open(os.path.join(feat_path, "galleryGANDescriptors.pkl"), "rb") as f:
                    galleryGANDescriptors = pickle.load(f)
                print("Gallery GAN features loaded successfully\n")
            except Exception as e:
                print(f"Error loading gallery GAN features: {e}")
                raise
        else:
            gan_query_loader = get_GAN_loader(gan_query_list, args.height, args.width, args.batch_size, args.workers)
            gan_gallery_loader = get_GAN_loader(gan_gallery_list, args.height, args.width, args.batch_size, args.workers)

            assert len(gan_query_loader.dataset) == len(gan_query_list)
            assert len(gan_gallery_loader.dataset) == len(gan_gallery_list) 
            
            print("[QUERY GAN EXTRACTION]\n")
            gan_feats_query, _ = extract_features(trainer.id_net, gan_query_loader)
            print("Query estratte correttamente\n")

            print("\n[GALLERY GAN EXTRACTION]\n")
            gan_feats_gallery, _ = extract_features(trainer.id_net, gan_gallery_loader)
            print("Gallery estratte correttamente\n")

            #Reshape + save locally
            queryGANDescriptors = torch.stack([
                torch.stack([gan_feats_query[p] for p in paths], dim=0)
                for paths in queryGANPaths
            ], dim=0)

            with open(os.path.join(feat_path, "queryGANDescriptors.pkl"), "wb") as f:
                pickle.dump(queryGANDescriptors, f)

            galleryGANDescriptors = torch.stack([
                torch.stack([gan_feats_gallery[p] for p in paths], dim=0)
                for paths in galleryGANPaths
            ], dim=0)

            with open(os.path.join(feat_path, "galleryGANDescriptors.pkl"), "wb") as f:
                pickle.dump(galleryGANDescriptors, f)

        '''
        # QUERY
        flat_query_feat = np.zeros((N * Q, D))
        for i, (p, _, _) in enumerate(tqdm(gan_query_list, desc="Building QUERY GAN descriptors")):
            flat_query_feat[i] = gan_feats_query[p]

        queryGANDescriptors = flat_query_feat.reshape(N, Q, D)
        
        # GALLERY
        flat_gallery_feat = np.zeros((M * Q, D))
        for i, (p, _, _) in enumerate(tqdm(gan_gallery_list, desc="Building GALLERY GAN descriptors")):
            flat_gallery_feat[i] = gan_feats_gallery[p]

        galleryGANDescriptors = flat_gallery_feat.reshape(M, Q, D)
        '''

        #assert queryGANDescriptors.shape == (N, 7, D)
        #assert len(queryGANPaths) == N
        #assert len(queryGANPaths[0]) == 7

        print("GalleryGAN:")
        print("  paths:", len(galleryGANPaths))
        print("  ids:  ", len(galleryGANIds))
        print("  cams: ", len(galleryGANCams))
        print("  descriptors:", galleryGANDescriptors.shape)
        #print(galleryGANPaths[:3])

        print("\nQuery:")
        print("  paths:", len(queryGANPaths))
        print("  ids:  ", len(queryGANIds))
        print("  cams: ", len(queryGANCams))
        print("  descriptors:", queryGANDescriptors.shape)

        print('\n\n')

        separate_camera_set = args.separate
        #separate_camera_set = True
        ranker = Ranker(galleryDescriptors, galleryCams, method, separate_camera_set)

        K = 3
        _, _, D = galleryGANDescriptors.shape
        filteredGalleryDescriptors = np.zeros((M, D))
        filteredGalleryGANDescriptors = np.zeros((M, K, D))
        indexGanSaved = np.zeros((M,K),dtype=np.int64)
        distGanSaved = np.zeros((M,K))

        for i in range(M):
            #filteredGalleryGANDescriptors[i], indexGanSaved[i] = select_topk_expansions(galleryDescriptors[i],galleryGANDescriptors[i],k=K)
            #filteredGalleryGANDescriptors[i] = filteredGalleryGANDescriptors[i].mean(axis=0)
            topk_feats, topk_idx, topk_dist = select_topk_expansions(galleryDescriptors[i],galleryGANDescriptors[i],k=K)

            filteredGalleryGANDescriptors[i] = topk_feats      
            indexGanSaved[i] = topk_idx                    
            distGanSaved[i] = topk_dist     
            filteredGalleryDescriptors[i] = topk_feats.mean(axis=0)

        for i in range(3):
            print(f"Image {i}:")
            for j, idx in enumerate(indexGanSaved[i]):
                angle = ANGLES[idx]
                dist  = distGanSaved[i][j]
                print(f"  rank {j+1}: angle = {angle:>3}°, dist = {dist:.4f}")
            print()

        dist_mat = np.zeros((N, M, 5))
        #weight = np.array([1.0] + [0.2] * len(ANGLES))
        #weight = np.array([1.0] + [0.2] * K)

        eps = 1e-6
        gan_weights = 1.0 / (topk_dist + eps)
        weight = np.concatenate(([1.0], gan_weights))
        weight = weight / weight.sum()

        for index in tqdm(range(N)):
            #print("evaluating image " + str(index) + " using " + method + " method ")
            #print(queryPaths[index])
            #print(queryGANPaths[index,:])
            queryDescriptor = queryDescriptors[index,:]

            #Classico senza expansion della query o della gallery
            #[distances, rank] = ranker.get_dist_rank(queryDescriptors[index,:], queryCams[index], [], [])
            [distances, rank] = ranker.get_dist_rank(queryDescriptors[index,:], queryCams[index])
            dist_mat[index,:,0] = distances

            #Calcola la distanza tra l'immagine originale e le immagini generate e prende le tre piu' vicine
            filteredQueryGan, _, _ = select_topk_expansions(queryDescriptors[index],queryGANDescriptors[index],k=3)

            queryDescriptorQE = np.concatenate((queryDescriptors[index][None, :], filteredQueryGan),axis=0)

            #queryDescriptorQE = np.concatenate((np.expand_dims(queryDescriptor,axis=0),np.squeeze(queryGANDescriptors[index, :, :])))

            #Con expansion della sola query
            [distances, rank] = ranker.QE(queryDescriptorQE, queryCams[index])
            dist_mat[index,:,1] = distances

            #puoi anche specificare un peso
            [distances, rank] = ranker.QE(queryDescriptorQE, queryCams[index], weight=weight)
            dist_mat[index,:,2] = distances

            #stessa cosa con expansion di query e gallery
            #[distances, rank] = ranker.GQE(queryDescriptorQE, queryCams[index], extGallery=galleryGANDescriptors)
            [distances, rank] = ranker.GQE(queryDescriptorQE, queryCams[index], extGallery=filteredGalleryGANDescriptors)
            dist_mat[index,:,3] = distances

            #[distances, rank] = ranker.GQE(queryDescriptorQE, queryCams[index], extGallery=galleryGANDescriptors, weight=weight)
            [distances, rank] = ranker.GQE(queryDescriptorQE, queryCams[index], extGallery=filteredGalleryGANDescriptors, weight=weight)
            dist_mat[index,:,4] = distances

        rankerExtended = Ranker(galleryDescriptors, galleryCams, method, separate_camera_set)
        dist_mat_extended = np.zeros((N, M, 5))

        #evaluation with fixed weight 
        fixed_weight = np.array([1.0] + [0.2] * len(ANGLES))
    
        for index in tqdm(range(N)):
            #print("evaluating image " + str(index) + " using " + method + " method ")
            #print(queryPaths[index])
            #print(queryGANPaths[index,:])
            queryDescriptor = queryDescriptors[index,:]

            #Classico senza expansion della query o della gallery
            #[distances, rank] = ranker.get_dist_rank(queryDescriptors[index,:], queryCams[index], [], [])
            [distances, rank] = rankerExtended.get_dist_rank(queryDescriptors[index,:], queryCams[index])
            dist_mat_extended[index,:,0] = distances

            #Calcola la distanza tra l'immagine originale e le immagini generate e prende le tre piu' vicine
            #filteredQueryGan, _, _ = select_topk_expansions(queryDescriptors[index],queryGANDescriptors[index],k=3)

            queryDescriptorQE = np.concatenate((queryDescriptors[index][None, :], filteredQueryGan),axis=0)

            queryDescriptorQE = np.concatenate((np.expand_dims(queryDescriptor,axis=0),np.squeeze(queryGANDescriptors[index, :, :])))

            #Con expansion della sola query
            [distances, rank] = rankerExtended.QE(queryDescriptorQE, queryCams[index])
            dist_mat_extended[index,:,1] = distances

            #puoi anche specificare un peso
            [distances, rank] = rankerExtended.QE(queryDescriptorQE, queryCams[index], weight=fixed_weight)
            dist_mat_extended[index,:,2] = distances

            #stessa cosa con expansion di query e gallery
            [distances, rank] = rankerExtended.GQE(queryDescriptorQE, queryCams[index], extGallery=galleryGANDescriptors)
            #[distances, rank] = ranker2.GQE(queryDescriptorQE, queryCams[index], extGallery=filteredGalleryGANDescriptors)
            dist_mat_extended[index,:,3] = distances

            [distances, rank] = rankerExtended.GQE(queryDescriptorQE, queryCams[index], extGallery=galleryGANDescriptors, weight=fixed_weight)
            #[distances, rank] = ranker2.GQE(queryDescriptorQE, queryCams[index], extGallery=filteredGalleryGANDescriptors, weight=weight)
            dist_mat_extended[index,:,4] = distances

        names = ["Baseline","QE","QE + weight","GQE","GQE + weight"]

        if separate_camera_set:
            print('\nSeparate camera set\n')
        else:
            print('\nNot separate camera set\n')

        print(method)

        print('\nWeight using distance and top 3 nearest GAN\n')
        for k, name in enumerate(names):
            print(f"\n=== {name} ===")
            evaluate_all(dist_mat[:, :, k], query_ids=queryIds, gallery_ids=galleryIds, query_cams=queryCams, gallery_cams=galleryCams, \
                cmc_topk=(1, 5, 10, 20), cmc_flag=True, separate_camera_set=separate_camera_set)
        

        print('\nFixed weight with all gan images\n')
        for k, name in enumerate(names):
            print(f"\n=== {name} ===")
            evaluate_all(dist_mat_extended[:, :, k], query_ids=queryIds, gallery_ids=galleryIds, query_cams=queryCams, gallery_cams=galleryCams, \
                cmc_topk=(1, 5, 10, 20), cmc_flag=True, separate_camera_set=separate_camera_set)
            
    else:
        raise NotImplementedError

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MV Training")
    # data
    parser.add_argument('--dataset-target', type=str, default='market1501', choices=datasets.names())
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")

    # model
    parser.add_argument('-a', '--arch', type=str, default='ft_net', choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)

    # training configs
    parser.add_argument('--init', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--idnet-fix", action="store_true")
    parser.add_argument('--stage', type=int, default=1)

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--mesh-dir', type=str, metavar='PATH', default='./examples/mesh/market/')

    # gan config
    parser.add_argument('--config', type=str, default='configs/latest.yaml', help='Path to the config file.')
    # parser.add_argument('--output_path', type=str, default='/outputs', help="generated images saving path")
    #parser.add_argument('--mode', type=str, default='recon')
    #parser.add_argument('--degree', type=int, default=45)

    #parser.add_argument('--gen', type=bool, default=False, help='Generate new query')
    parser.add_argument('--gen', type=int, default=0, help='Generate new query')
    parser.add_argument('--separate', type=bool, default=False, help='Separate camera set')
    #un'opzione per poter recuperare le feature gia' estratte o generarle nuove
    parser.add_argument('--lf', action='store_true', help='Local feature, recupera localmente le feature')

    main()
