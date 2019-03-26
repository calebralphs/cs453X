import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import math

def fPC (y, yhat):
    return np.count_nonzero(np.equal(y, np.greater(yhat, [.5] * len(y)))) / len(y)

# decidied to use alternate method of measuring accuracy of predictors becuase this was unefficient
def measureAccuracyOfPredictors (predictors, X, y):
    featurePermLabels = []
    for pred in predictors:
        featurePermLabels.append(X[:, pred[0]] > X[:, pred[1]])
    return fPC(y, np.sum(featurePermLabels, axis = 0) / len(predictors))

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    vectTrainingFaces = vectorizeFaceMatrix(trainingFaces) # vectorized training faces
    vectTestingFaces = vectorizeFaceMatrix(testingFaces) # vectorized testing faces
    # permutations of pixel pair idx
    pixelIdxPerms = np.array(np.meshgrid(list(range(576)), list(range(576)))).T.reshape(-1,2)
    
    # arrays to keep track of information on the best features for efficiency
    bestFeatures = []
    bestFeature_fPCs = []
    bestFeaturePermLabels = []
    bestPixels = []
    
    for m in range(5):
        # array to keep track of feature's fPCs
        predFeature_fPCs = []
        for pixelPairIdx in pixelIdxPerms:
            permLabels = vectTrainingFaces[:, pixelPairIdx[0]] > vectTrainingFaces[:, pixelPairIdx[1]]
            testingPermLabels = np.sum(bestFeaturePermLabels + [permLabels], axis = 0) / (m + 1)
            predFeature_fPCs.append(fPC(trainingLabels, testingPermLabels))
        
        bestFeatureIdx = np.argmax(predFeature_fPCs)
        bestFeature = pixelIdxPerms[bestFeatureIdx]
        pixelIdxPerms = np.delete(pixelIdxPerms, bestFeatureIdx, axis = 0)
        bestFeature_fPC = predFeature_fPCs[bestFeatureIdx]
        bestFeaturePermLabel = vectTrainingFaces[:, bestFeature[0]] > vectTrainingFaces[:, bestFeature[1]]
        
        bestFeatures.append(bestFeature)
        bestFeature_fPCs.append(bestFeature_fPC)
        bestFeaturePermLabels.append(bestFeaturePermLabel)
        r1, c1, r2, c2 = pixelIdxFromVectIdx(bestFeature)
        print(m, bestFeature, [r1, c1, r2, c2], bestFeature_fPC)
        bestPixels.append([r1, c1, r2, c2])
        
    show = True
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        colors = ['r', 'b', 'y', 'g', 'm']
        for i in range(5):
            r1, c1, r2, c2 = bestPixels[i]
            # Show r1,c1
            rect = patches.Rectangle((c1,r1),1,1,linewidth=2,edgecolor=colors[i],facecolor='none')
            ax.add_patch(rect)
            # Show r2,c2
            rect = patches.Rectangle((c2,r2),1,1,linewidth=2,edgecolor=colors[i],facecolor='none')
            ax.add_patch(rect)
        # Display the merged result
        plt.show() 
    print(bestFeatures, bestFeature_fPCs)
    return measureAccuracyOfPredictors(bestFeatures, vectTestingFaces, testingLabels)
            
def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

def vectorizeFaceMatrix(faces):
    return faces.reshape(-1, 576)

def pixelIdxFromVectIdx(idxPair):
    return math.floor(idxPair[0] / 24), math.floor(idxPair[0] % 24), math.floor(idxPair[1] / 24), math.floor(idxPair[0] % 24)

def sampleTestingAccuracy(trainingFaces, trainingLabels, testingFaces, testingLabels):
    sampleTestingAccuracies = []
    for n in np.array(list(range(1, 6))) * 400:
        sampleAccuracy = stepwiseRegression(trainingFaces[:n], trainingLabels[:n], testingFaces[:n], testingLabels[:n])
        sampleTestingAccuracies.append(sampleAccuracy)
        print(n, sampleAccuracy)
    return sampleTestingAccuracies
if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    testingAccuracy = stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)
    sampleTestingAccuracies = sampleTestingAccuracy(trainingFaces, trainingLabels, testingFaces, testingLabels)