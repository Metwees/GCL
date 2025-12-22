from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from scipy.spatial import distance
import numpy as np


class Ranker:
    #Ho rimosso datasetPath
    def __init__(self, gallery, cams, method, separate_camera_set):
        self.gallery = gallery #gallery images featureSet
        self.cams = cams #gallery images cameras
        self.method = method # RF method used for re-ranking
        self.separate_camera_set = separate_camera_set #separate gallery sets


    def get_separate_rank(self, distance, rank, cam):
        if self.separate_camera_set: # Filter out samples from same camera
            sameCameraSet = self.cams == cam
            otherCameraSet = self.cams != cam
            distance[sameCameraSet] = np.inf
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
    def get_dist_rank(self, query, cam):
        distances = euclidean_distances(np.expand_dims(query, axis=0), self.gallery)
        rank = np.argsort(distances)
        return self.get_separate_rank(distances[0], rank[0], cam)
      

    #Exploit query expansion
    def QE(self, queries, cam, weight=None):
        """
        Query Expansion: usa più query (es. top-k neighbors) per ottenere un'unica distanza verso la gallery.
        - queries: np.ndarray (K, D)
        - weight:  None oppure array di shape (K,) con pesi per le query
        """
        print("QE")
        if "center" not in self.method:
            distances = self.get_weighted_distance(self.gallery, queries, weight)  # (N, K)
            if "min" in self.method:
                distance = distances.min(axis=1)
            elif "avg" in self.method:
                distance = distances.mean(axis=1)
        else:
            if weight is None:
                mean_query = np.mean(queries, axis=0)
            else:
                w = np.asarray(weight, dtype=float)
                mean_query = np.average(queries, axis=0, weights=w)
            distance = euclidean_distances(np.expand_dims(mean_query, axis=0), self.gallery)[0]
        rank = np.argsort(distance)
        return self.get_separate_rank(distance, rank, cam)


    #Exploit gallery and query expansion
    def GQE(self, queries, cam, extGallery, weight=None):
        """
        Gallery + Query Expansion:
        - queries:     (K, D)
        - extGallery:  (N_gallery, M, D) M espansioni/varianti per ciascun item di gallery
        - weight:      None oppure (K,) pesi per le query
        """
        print("GQE")
        if "center" not in self.method:
            # calculate the distances between queries and gallery
            distance = self.get_weighted_distance(self.gallery, queries, weight)  # (N, K)
            distances = np.expand_dims(distance, axis=0)
            for i in range(np.shape(extGallery)[1]):
                distance = self.get_weighted_distance(extGallery[:,i,:], queries, weight)
                distances = np.concatenate((distances, np.expand_dims(distance, axis=0)),  axis=0)
            print(np.shape(distances))
            
            if "min" in self.method:
                # Best tra le espansioni e tra le query
                # prima riduciamo sulle espansioni, poi sulle query
                distances_over_gallery = distances.min(axis=0)  # (N, K)
                distance = distances_over_gallery.min(axis=1)   # (N,)
            elif "avg" in self.method:
                distances_over_gallery = distances.mean(axis=0)  # (N, K)
                if weight is not None:
                    distance = distances_over_gallery.sum(axis=1)  # (N,)
                else:
                    distance = distances_over_gallery.mean(axis=1)  # (N,)(axis=1)
            print(np.shape(distance))
        else:
            if weight is None:
                mean_query = np.mean(queries, axis=0)
                # print("mean query", np.shape(mean_query))
                mean_gallery = np.mean(np.concatenate((np.expand_dims(self.gallery,axis=1), extGallery),  axis=1), axis=1)
                # print("mean gallery", np.shape(mean_gallery))
            else:
                w = np.asarray(weight, dtype=float)
                mean_query = np.average(queries, axis=0,  weights=weight)
                # print("mean query", np.shape(mean_query))
                # print(np.shape(np.expand_dims(self.gallery,axis=1)))
                # print(np.shape(extGallery))
                mean_gallery = np.average(np.concatenate((np.expand_dims(self.gallery,axis=1), extGallery),  axis=1), axis=1)
                # print("mean gallery", np.shape(mean_gallery))
            distance = euclidean_distances(np.expand_dims(mean_query, axis=0), mean_gallery)[0]
        rank = np.argsort(distance)
        return self.get_separate_rank(distance, rank, cam)


    def get_weighted_distance(self, gallery, queries, weight=None):
        """
        Distanze euclidee tra ogni item della gallery e ciascuna query.
        - gallery: (N, D)
        - queries: (K, D)
        Ritorna: (N, K)
        """
        distances = euclidean_distances(gallery, queries)  # (N, K)
        if weight is not None:
            w = np.asarray(weight, dtype=float)
            w = w / (w.sum() + 1e-12)
            # Applica i pesi sulle colonne (asse delle query)
            distances = distances * w  # broadcasting: (N, K) * (K,)
        return distances