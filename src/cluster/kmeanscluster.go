/*
kmeanscluster.go
Cluster data using the K-Means algorithm.  Initialize the Radial Basis Function (RBF) centroids by selecting data
points farthest from one another in Euclidean distance. Iterate the following two
steps until the fractional within-class squared sum (WCSS) delta is less than a threshold or max iterations:
1.  Place the data in the cluster whose centroid is closest in Euclidean distance
2.  Recalculate the cluster centroids using the points assigned to that cluster
*/

package cluster

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path"
)

const (
	tol           = 0.001         // K iteration stopping criteria
	dataDir       = "../rbf/data" // directory for the weights and synthetic speech files
	kmeans        = "kmeans.csv"  // file for k-means
	maxIterations = 100           // maximum iterations for reassigning clusters and updating centroids
)

type Cluster struct {
	centroids   [][]float64 // mean of the Radial Basis Function
	wcss        []float64   // within class squared sum
	bw          []float64   // bandwidth of the RBF
	dataCluster []int       // the cluster the data is assigned to
}

// Create a new K-means Cluster object
func newCluster(nCentroids int, dataLen int, centrDim int) *Cluster {
	centroids := make([][]float64, nCentroids)
	for i := range centroids {
		centroids[i] = make([]float64, centrDim)
	}
	bw := make([]float64, nCentroids)
	wcss := make([]float64, nCentroids)
	dataCluster := make([]int, dataLen)
	return &Cluster{
		centroids:   centroids,
		bw:          bw,
		wcss:        wcss,
		dataCluster: dataCluster,
	}
}

// Recalculate the centroids by averaging data
func (cl *Cluster) updateCentroids(data [][]float64, centr int, dataLen int, centrDim int) error {

	// loop over centroids
	for m := 0; m <= centr; m++ {
		sum := make([]float64, centrDim)
		count := 0
		if centr == 63 {
			fmt.Printf("centroid=%d, ", m)
		}
		// loop over data in this cluster
		for n := range dataLen {
			if cl.dataCluster[n] == m {
				for i := range sum {
					sum[i] += data[n][i]
				}
				count++
			}
		}
		if centr == 63 {
			fmt.Printf("count=%d\n", count)
		}
		// new centroid location
		for i := range cl.centroids[m] {
			cl.centroids[m][i] = sum[i] / float64(count)
		}
	}
	return nil
}

// Compute a common bandwidth for the RBFs
func (cl *Cluster) computeRBFbandwidth(ncentroids int) {
	// overfit if bw too small, underfit if too large
	// Try these for bandwidth calculation
	// 1. use the average distance between all the centroids divided by two
	// 2. use max distance between centroids divided by the square root of twice the number of centroids
	//sum := 0.0
	//count := 0
	max := 0.0
	distance := 0.0
	for i := 0; i < ncentroids-1; i++ {
		for j := i + 1; j < ncentroids; j++ {
			distance = cl.distance(cl.centroids[i], cl.centroids[j])
			//sum += distance
			if distance > max {
				max = distance
			}
			//count++
		}
	}
	//bandwidth := 0.5 * sum / float64(count)
	bandwidth := max / math.Sqrt(float64(2*ncentroids))
	// assign the bandwidth to clusters
	for i := range ncentroids {
		cl.bw[i] = bandwidth
	}
	fmt.Printf("\nbandwidth=%.2f\n", bandwidth)
}

// Reassign clusters to the nearest centroid in Euclidean distance
func (cl *Cluster) reassignClusters(data [][]float64, centr int, iters int, dataLen int) (float64, error) {
	wcss := 0.0

	for n := range dataLen {
		mindist := math.MaxFloat64
		for m := 0; m <= centr; m++ {
			dist := cl.distance(data[n], cl.centroids[m])
			if dist < mindist {
				mindist = dist
				cl.dataCluster[n] = m
			}
		}
		wcss += mindist * mindist
	}

	// stop iterations if not changing
	delta := math.Abs(wcss-cl.wcss[centr-1]) / cl.wcss[centr-1]
	if delta < tol || iters == maxIterations {
		return wcss, fmt.Errorf("stop iterations")
	}
	return wcss, nil
}

// Create a new centroid farthest away in Euclidean distance from other centroids
func (cl *Cluster) newCentroid(data [][]float64, centr int, dataLen int) error {
	maxdist := 0.0
	mindata := 0
	maxdata := 0

	// Create the first centroid randomly from one of the data
	if centr == 0 {
		first := rand.Intn(dataLen)
		// assign the first centroid
		for i := range cl.centroids[centr] {
			cl.centroids[centr][i] = data[first][i]
		}
		wcss := 0.0
		for n := range dataLen {
			dist := cl.distance(data[n], cl.centroids[centr])
			wcss += dist * dist
		}
		cl.wcss[centr] = wcss
		return nil
	}

	// find the min distance for this data point to the current centroids
	// then find the max of the mins of all the data points
	for n := range dataLen {
		// loop over the current centroids
		mindist := math.MaxFloat64
		for m := range centr {
			dist := cl.distance(data[n], cl.centroids[m])
			if dist < mindist {
				mindist = dist
				mindata = n
			}
		}
		if mindist > maxdist {
			maxdist = mindist
			maxdata = mindata
		}
	}

	// assign the next centroid using the max(min) from above
	for i := range cl.centroids[centr] {
		cl.centroids[centr][i] = data[maxdata][i]
	}
	return nil
}

// Euclidean distance between two points in data space
func (cl *Cluster) distance(data []float64, centroid []float64) float64 {
	sum := 0.0
	for i := range data {
		diff := data[i] - centroid[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// Save K-Means Cluster data
func (cl *Cluster) saveClusterData() error {
	// save to disk centroids, bandwidths, wcss
	fout, err := os.Create(path.Join(dataDir, kmeans))
	if err != nil {
		fmt.Printf("file create %s error: %v\n", kmeans, err.Error())
		return fmt.Errorf("file create %s error: %v", kmeans, err.Error())
	}
	defer fout.Close()
	for i := range cl.centroids {
		for _, val := range cl.centroids[i] {
			fmt.Fprintf(fout, "%.16f,", val)
		}
		fmt.Fprintf(fout, "%.16f,%.16f\n", cl.bw[i], cl.wcss[i])
	}
	return nil
}

// Cluster the data using the K-Means algorithm for ncentroids
func Kmeans(ncentroids int, data [][]float64) error {
	// Create Cluster object
	// number of input vectors
	dataLen := len(data)
	// centroid dimension is the same as the input vector length
	centrDim := len(data[0])
	kmc := newCluster(ncentroids, dataLen, centrDim)
	centr := 0

	// Create the first centroid, and use the same starting
	// point for all the trials below
	err := kmc.newCentroid(data, centr, dataLen)
	if err != nil {
		fmt.Printf("newCentroid %d error: %v\n", centr, err.Error())
		return fmt.Errorf("newCentroid %d error: %v", centr, err.Error())
	}

	// create the remaining centroids and assign data to the clusters
	for centr = 1; centr < ncentroids; centr++ {

		// Create centr new centroids, don't reuse old ones
		for n := range centr {
			err := kmc.newCentroid(data, n+1, dataLen)
			if err != nil {
				fmt.Printf("newCentroid %d error: %v\n", n, err.Error())
				return fmt.Errorf("newCentroid %d error: %v", n, err.Error())
			}
		}

		iters := 1
		// interate until fractional wcss delta between iterations is less than tolerance
		for {
			// place the data in nearest cluster based on Euclidean distance
			if wcss, err := kmc.reassignClusters(data, centr, iters, dataLen); err != nil {
				// stop iterating for this centr
				kmc.wcss[centr] = wcss
				break
			}

			// find new centroids by averaging data in each cluster
			kmc.updateCentroids(data, centr, dataLen, centrDim)
			iters++
		}
	}

	// compute common bandwidth for all RBFs
	kmc.computeRBFbandwidth(ncentroids)

	// save to disk centroids, bandwidths, wcss
	err = kmc.saveClusterData()
	if err != nil {
		fmt.Printf("saveClusterData error: %v\n", err.Error())
		return fmt.Errorf("saveClusterData error: %v", err.Error())
	}
	return nil
}
