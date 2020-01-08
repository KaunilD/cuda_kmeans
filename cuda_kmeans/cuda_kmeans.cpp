// cuda_kmeans.cpp : Defines the entry point for the application.
//

#include "cuda_kmeans.h"

using std::cout;
using std::endl;

using DataSet = std::vector<Point>;

double square(double v) {return v * v;}

double l2Norm(Point p1, Point p2) {
	return sqrt(square(p1.x - p2.x) + square(p1.y - p2.y));
}


DataSet kMeans(const DataSet& data, size_t k, size_t iterations) {
	/*
		Step 1:
			Assign random cluster centers to the 
			DataSet. 
	*/

	// initialize the random number generator.
	// Initializing seed for pseudo-randomness
	// replace with default_random_engine in prod
	static std::random_device seed;
	static std::mt19937 random_number_generator(seed()); // 
	// PDF for uniform distribution
	std::uniform_int_distribution<size_t> pdf(0, data.size() - 1);

	// stores actual cluster_centers
	DataSet cluster_centers(k);
	for (auto center : cluster_centers) {
		center = data[pdf(random_number_generator)];
	}

	// contains index of cluster center in cluster_centers 
	// for each data point in the dataset.
	std::vector<size_t> cluster_assignments(data.size());

	/*
		Step 2:
			Find cluster assignments to each data point
			by iterative expectation maximization
	*/
	for (size_t iteration = 0; iteration < iterations; ++iteration) {
		
		// Step 2.1: find assignments
		// for each point, measure the distance from all the cluster_centers.
		for (size_t p_idx = 0; p_idx < data.size(); ++p_idx) {
			double shortest_distance = std::numeric_limits<double>::max();
			size_t best_cluster = 0;
			for (size_t center_idx = 0; center_idx < cluster_centers.size(); ++center_idx) {
				double distance = l2Norm(data[p_idx], cluster_centers.at(center_idx));
				if (distance < shortest_distance) {
					shortest_distance = distance;
					best_cluster = center_idx;
;				}
			}
			cluster_assignments[p_idx] = best_cluster;
		}

		// Step 2.2
		// calculate new means from assigned clusters
		DataSet new_cluster_centers(k);

		// get cluster histogram and new cluster sum
		std::vector<size_t> counts(k, 0); // store counts of each cluster assignment
		for (size_t p_idx = 0; p_idx < data.size(); p_idx++) {
			const auto cluster_idx = cluster_assignments[p_idx];
			counts[cluster_idx] += 1;
			new_cluster_centers[cluster_idx].x += data[p_idx].x;
			new_cluster_centers[cluster_idx].y += data[p_idx].y;
		}

		for (size_t idx = 0; idx < k; ++idx) {
			const auto count = std::max<size_t>(1, counts.at(idx));

			cluster_centers[idx].x = new_cluster_centers[idx].x / count;
			cluster_centers[idx].y = new_cluster_centers[idx].y / count;
		}

	}

	return cluster_centers;
}


int main()
{
	DataSet dataset{
		Point{1, 2}, Point{1, 4}, Point{1, 0},
		Point{10, 2}, Point{10, 4}, Point{10, 0}
	};
	int k = 2, n_iter = 100;

	DataSet centers = kMeans(dataset, k, n_iter);

	cout << centers[0].x << ", " << centers[0].y << endl;
	cout << centers[1].x << ", " << centers[1].y << endl;
	while (true) {

	}
	return 0;
}
