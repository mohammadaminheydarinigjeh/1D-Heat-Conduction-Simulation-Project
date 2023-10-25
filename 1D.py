import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Define the domain and parameters
L = 1.0  # Length of the 1D domain
num_elements = 10  # Number of elements
num_nodes = num_elements + 1  # Number of nodes
dx = L / num_elements  # Element size

# Define the source term and boundary conditions
def source_term(x):
    return 1.0

def boundary_condition(x):
    return 0.0

# Create the mesh (node coordinates)
nodes = np.linspace(0, L, num_nodes)
elements = np.array([list(range(i, i+2)) for i in range(num_elements)])

# Initialize the stiffness matrix and load vector
K = lil_matrix((num_nodes, num_nodes))
F = np.zeros(num_nodes)

# Define a function for the element stiffness matrix
def element_stiffness_matrix(e):
    x1, x2 = nodes[elements[e]]
    k = np.zeros((2, 2))
    k[0, 0] = 1 / dx
    k[0, 1] = -1 / dx
    k[1, 0] = -1 / dx
    k[1, 1] = 1 / dx
    return k

# Define a function for the element load vector
def element_load_vector(e):
    x1, x2 = nodes[elements[e]]
    f = np.zeros(2)
    f[0] = 0.5 * dx * source_term((x1 + x2) / 2)
    f[1] = 0.5 * dx * source_term((x1 + x2) / 2)
    return f

# Assemble the stiffness matrix and load vector
for e in range(num_elements):
    ke = element_stiffness_matrix(e)
    fe = element_load_vector(e)
    for i in range(2):
        for j in range(2):
            K[elements[e, i], elements[e, j]] += ke[i, j]
        F[elements[e, i]] += fe[i]

# Apply boundary conditions
for i in [0, num_nodes - 1]:
    K[i, :] = 0
    K[i, i] = 1
    F[i] = boundary_condition(nodes[i])

# Convert the stiffness matrix to a compressed sparse row (CSR) matrix
K = csr_matrix(K)

# Solve the linear system using SciPy's sparse linear solver
U = spsolve(K, F)

# Print the solution at the nodes
for i, u in enumerate(U):
    print(f"Node {i}: U = {u:.4f}")