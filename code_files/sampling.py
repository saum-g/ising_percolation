from random_graph import conn_prob_grid
import numpy as np

if __name__ == "__main__":

	num_uniform_pts = 50

	uniform_points = np.linspace(0, 0.5, num_uniform_pts, endpoint = False)

	transition_pt = 1/2-1/(2*np.sqrt(2))

	# Dense points around c
	dense_points = np.linspace(transition_pt - 0.01, transition_pt + 0.01, 20)

	# Combine and remove duplicates
	combined_ps = np.union1d(uniform_points, dense_points)

	print("combined_ps=")
	print(combined_ps)

	p_mat = (1-2*combined_ps)**2
	print("p_mat=")
	print(p_mat)

	dim_vals = {}

	for dim in range(10, 101):
		print("doing dim=", dim)

		means = []
		stds = []

		for p in p_mat:
		    hor_grid = p*np.ones((dim, dim))
		    ver_grid = p*np.ones((dim, dim))


		    m,s = conn_prob_grid(dim, hor_grid, ver_grid, 0, dim/2*(1+dim), 10000)
		    means.append(float(m))
		    stds.append(float(s))

		    # print(m,s)

		dim_vals[dim] = (means, stds)


		print(dim_vals)



	print(dim_vals)



	


