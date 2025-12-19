from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from scipy.spatial import distance
import numpy as np
import sys, os, re


class Ranker:
    #Ho rimosso datasetPath
    def __init__(self, gallery, cams, method, separate_camera_set, eng=None):
        self.gallery = gallery #gallery images featureSet
        self.cams = cams #gallery images cameras
        self.method = method # RF method used for re-ranking
        self.separate_camera_set = separate_camera_set #separate gallery sets

    def get_separate_rank(self, distance, rank, cam):
        if self.separate_camera_set: # Filter out samples from same camera
            sameCameraSet = self.cams == cam
            otherCameraSet = self.cams != cam
            distance[sameCameraSet] = np.max(distance)
            #rank = np.hstack((rank[otherCameraSet], rank[sameCameraSet] ))
            rank = np.concatenate((rank[otherCameraSet], rank[sameCameraSet]), axis=0)
        return (distance, rank)


    # Function to calculate the distances and ranking without expansions
    # INPUT
    #   query   : query image features
    #   cam     : query image camera
    # OUTPUT
    #   distances  : gallery image distances
    #   rank : index of the sorted (by distance) gallery element
    def get_dist_rank(self, query, cam, ):
        distances = euclidean_distances(np.expand_dims(query, axis=0), self.gallery)
        rank = np.argsort(distances)
        return self.get_separate_rank(distances[0], rank[0], cam)
      

    #Exploit query expansion
    def QE(self, queries, cam, weight=None):
        print("QE")
        if "center" not in self.method:
            # calculate the distances between queries and gallery
            distances = self.get_weighted_distance(self.gallery, queries, weight)
            if "min" in self.method:
                # extract for each image the closest image distances
                distance = distances.min(axis=1)
            elif "avg" in self.method:
                # extract for each image the average image distances
                distance = distances.mean(axis=1)
        else:
            if weight is None:
                mean_query = np.mean(queries, axis=0)
            else:
                mean_query = np.average(queries, axis=0, weights=weight)
            distance = euclidean_distances(np.expand_dims(mean_query, axis=0), self.gallery)[0]
        rank = np.argsort(distance)
        return self.get_separate_rank(distance, rank, cam)


    #Exploit gallery and query expansion
    def GQE(self, queries, cam, extGallery, weight=None):
        print("GQE")
        if "center" not in self.method:
            # calculate the distances between queries and gallery
            distance = self.get_weighted_distance(self.gallery, queries, weight)
            distances = np.expand_dims(distance, axis=0)
            for i in range(np.shape(extGallery)[1]):
                distance = self.get_weighted_distance(extGallery[:,i,:], queries, weight)
                distances = np.concatenate((distances, np.expand_dims(distance, axis=0)),  axis=0)
            print(np.shape(distances))
            if "min" in self.method:
                distances = distances.min(axis=0)
                distance = distances.min(axis=1)
            elif "avg" in self.method:
                distances = distances.mean(axis=0)
                distance = distances.mean(axis=1)
            print(np.shape(distance))
            rank = np.argsort(distance)
        else:
            if weight is None:
                mean_query = np.mean(queries, axis=0)
                # print("mean query", np.shape(mean_query))
                mean_gallery = np.mean(np.concatenate((np.expand_dims(self.gallery,axis=1), extGallery),  axis=1), axis=1)
                # print("mean gallery", np.shape(mean_gallery))
            else:
                mean_query = np.average(queries, axis=0,  weights=weight)
                # print("mean query", np.shape(mean_query))
                # print(np.shape(np.expand_dims(self.gallery,axis=1)))
                # print(np.shape(extGallery))
                mean_gallery = np.average(np.concatenate((np.expand_dims(self.gallery,axis=1), extGallery),  axis=1), axis=1, weights=weight)
                # print("mean gallery", np.shape(mean_gallery))
            distance = euclidean_distances(np.expand_dims(mean_query, axis=0), mean_gallery)[0]
            rank = np.argsort(distance)
        return self.get_separate_rank(distance, rank, cam)



    def get_weighted_distance(self, gallery, queries, weight=None):
        distances = euclidean_distances(self.gallery, queries)
        if weight is not None:
            distances = distances * weight
        return distances
