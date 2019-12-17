import numpy as np
import matplotlib.pyplot as plt

# Width of the vector field.
width = 100.0
# Number of cells to an edge.
n = 51
xs = np.linspace(0.0, width, n)
ys = np.linspace(0.0, width, n)
# Edge length of a cell.
h = width / n
xs, ys = np.meshgrid(xs, ys)
# First, generate a stream function.
ts = np.sin(xs / (0.1 * width)) * np.sin(ys / (0.1 * width))
# Second, generate a scalar potential.
ps = np.cos(xs / (0.2 * width)) * np.cos(ys / (0.2 * width))

# Put them together to make a vector field.
# Create staggered coordinates.
sxs = np.linspace(0.5 * h, width - 0.5 * h, n - 1)
sys = np.linspace(0.5 * h, width - 0.5 * h, n - 1)
sxs, sys = np.meshgrid(sxs, sys)
curl_kernel = (1 / (2 * h)) * np.array([
    [
        [-1, -1],
        [ 1,  1]
    ],
    [
        [1, -1],
        [1, -1]
    ]
])
grad_kernel = (1 / (2 * h)) * np.array([
    [
        [-1, 1],
        [-1, 1]
    ],
    [
        [-1, -1],
        [ 1,  1]
    ]
])
vts = np.zeros((2, *np.shape(sxs)))
vps = np.zeros((2, *np.shape(sxs)))
print("Generating vector field.")
for j in range(np.shape(vts)[1]):
    for i in range(np.shape(vts)[2]):
        vts[:,j,i] = np.sum(curl_kernel * ts[j:j+2,i:i+2], axis=(1,2))
        vps[:,j,i] = np.sum(grad_kernel * ps[j:j+2,i:i+2], axis=(1,2))
# The final vector field is the sum of an irrotational part and a solenoidal
# part.
vs = vts + vps

# Now use the Gauss-Seidel method to recover the solendoidal and irrotational
# parts.

laplacian_kernel = (1 / h**2) * np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
])
# This kernel is basically the important thing here. By using this kernel, the
# solenoidal part can be approached iteratively without making use of any
# intermediate "potential" type fields.
coeff_kernel = (1 / (4 * h**2)) * np.array([
    [
        [
            [ 1,  2,  1],
            [-2, -4, -2],
            [ 1,  2,  1]
        ],
        [
            [-1, 0,  1],
            [ 0, 0,  0],
            [ 1, 0, -1]
        ]
    ],
    [
        [
            [-1, 0,  1],
            [ 0, 0,  0],
            [ 1, 0, -1]
        ],
        [
            [1, -2, 1],
            [2, -4, 2],
            [1, -2, 1]
        ]
    ]
])

print("Finding solenoidal part iteratively.")
# We could start by copying the `vs` array to have a good initial guess, but
# since we are testing the algorithm here, let's start with a zeroed out field
# and see if the solenoidal part can be recovered.
# rts = np.copy(vs)
rts = np.zeros(np.shape(vs))
norm = laplacian_kernel[1,1]
# Set boundary parts to be the same.
for j in range(np.shape(rts)[1]):
    rts[:,j,0] = vs[:,j,0]
    rts[:,j,-1] = vs[:,j,-1]
for i in range(np.shape(rts)[2]):
    rts[:,0,i] = vs[:,0,i]
    rts[:,-1,i] = vs[:,-1,i]
for iter_count in range(1, 100):
    if iter_count % 10 == 0:
        print("Iteration #%d." % iter_count)
    for j in range(1, np.shape(vs)[1] - 1):
        for i in range(1, np.shape(vs)[2] - 1):
            laplacian = np.sum(laplacian_kernel * rts[:,j-1:j+2,i-1:i+2], axis=(1,2))
            coeff = np.sum(coeff_kernel * vs[:,j-1:j+2,i-1:i+2], axis=(1,2,3))
            rts[:,j,i] += 1 / norm * (coeff - laplacian)

# Analyze how good we did at getting the solenoidal part from the vector field.
print("Finding divergence of direct iterative solution.")
rdivs = np.zeros(np.shape(sxs))
vdivs = np.zeros(np.shape(sxs))
for j in range(1, np.shape(rdivs)[0] - 1):
    for i in range(1, np.shape(rdivs)[1] - 1):
        rdivs[j,i] = np.sum(grad_kernel * rts[:,j-1:j+1,i-1:i+1])
        vdivs[j,i] = np.sum(grad_kernel * vs[:,j-1:j+1,i-1:i+1])
print("RMS divergence of direct iterative solution: %f" % np.sqrt(np.mean(rdivs**2)))
print("RMS divergence of original vector field:     %f" % np.sqrt(np.mean(vdivs**2)))
print("Chi squared of direct iterative solution:         %f" % np.sqrt(np.mean((rts - vts)**2)))
print("Chi squared percent of direct iterative solution: %f%%" % np.sqrt(np.mean((100 * (rts / vts - 1))**2)))

print ("Finding potential iteratively.")
rps = np.zeros(np.shape(ps))
norm = laplacian_kernel[1,1]
for iter_count in range(1, 100):
    if iter_count % 10 == 0:
        print("Iteration #%d." % iter_count)
    for j in range(1, np.shape(rps)[0] - 1):
        for i in range(1, np.shape(rps)[1] - 1):
            laplacian = np.sum(laplacian_kernel * rps[j-1:j+2,i-1:i+2])
            coeff = vdivs[j-1,i-1]
            rps[j,i] += 1 / norm * (coeff - laplacian)
    # Apply boundary conditions.
    for j in range(np.shape(rps)[0]):
        rps[j,0] = 0
        rps[j,-1] = 0
    for i in range(np.shape(rps)[1]):
        rps[0,i] = 0
        rps[-1,i] = 0

# Get a second estimate of the solenoidal part from the potential.
rts2 = np.zeros(np.shape(vs))
for j in range(np.shape(rts2)[1]):
    for i in range(np.shape(rts2)[2]):
        rts2[:,j,i] = vs[:,j,i] - np.sum(grad_kernel * rps[j:j+2,i:i+2], axis=(1,2))
print("Chi squared of indirect iterative solution:         %f" % np.sqrt(np.mean((rts2 - vts)**2)))
print("Chi squared percent of indirect iterative solution: %f%%" % np.sqrt(np.mean((100 * (rts2 / vts - 1))**2)))

# Plotting.
extents = [
    np.min(xs) - 0.5 * h,
    np.max(xs) + 0.5 * h,
    np.min(ys) - 0.5 * h,
    np.max(ys) + 0.5 * h
]
plt.title("Vector field")
plt.quiver(sxs, sys, vs[0,:,:], vs[1,:,:])
plt.show()
plt.title("Irrotational part")
plt.imshow(ps, origin='lower', extent=extents)
plt.quiver(sxs, sys, vps[0,:,:], vps[1,:,:])
plt.show()
plt.title("Solenoidal part")
plt.imshow(ts, origin='lower', extent=extents)
plt.quiver(sxs, sys, vts[0,:,:], vts[1,:,:])
plt.show()
plt.title("Iterative solutions compared to solenoidal part")
plt.quiver(sxs, sys, vts[0,:,:], vts[1,:,:], color='k', label="solenoidal part")
plt.quiver(sxs, sys, rts2[0,:,:], rts2[1,:,:], color='b', label="indirect solution")
plt.quiver(sxs, sys, rts[0,:,:], rts[1,:,:], color='r', label="direct solution")
plt.legend()
plt.show()
plt.title("Divergence of iterative solution compared to irrotational part")
plt.imshow(rdivs, origin='lower', extent=extents)
plt.show()

