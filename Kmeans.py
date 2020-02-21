import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
data = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
                 [1, 3],
                 [8, 9],
                 [0, 3],
                 [5, 4],
                 [6, 4]
                 ])


colors = 10*["g","r","c","b","k"]
class K_Means:
    def __init__(self, k=4, tolerance=0.01, iterations=5): #initiliazor
        self.k = k
        self.tolerance = tolerance
        self.iterations = iterations

    def train(self, data):
        counter = 0
        adjusting = []
        # centroids adjusting
        self.centroids = {}        # adjusting the centriods
        for i in range(self.k):

            self.centroids[i] = data[i] #  assume the first point
        for i in range(self.iterations):
            print("iteration number : " , counter )
            counter = counter +1
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []
            for featureset in data:
                # Distances between every data point and the centroids are calculated and stored
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset) # b3ml classification ll 7aga el oryba el awl 3la 7sb a2l distance

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                #New cluster centroid positions are updated: similar to finding the mean
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
                plt.imshow(self.classifications[classification][1])
                plt.show()
                optimized = True

            for c in self.centroids:
                    original_centroid = prev_centroids[c]
                    current_centroid = self.centroids[c]
                    #BSHOF el movement 3ddt el tolerance wla l2a

                    if np.linalg.norm(current_centroid - original_centroid) > self.tolerance:
                        optimized = False
            if(      np.linalg.norm(current_centroid - original_centroid) >0) :
                print(np.linalg.norm(current_centroid - original_centroid))

            adjusting.append(np.linalg.norm(current_centroid - original_centroid))

            if optimized : # if it is smaller than the tolerance
               #         print(featureset)
                break


        return  adjusting




    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification





