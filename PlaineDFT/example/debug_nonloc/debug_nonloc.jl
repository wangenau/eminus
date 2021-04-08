using PyCall
scriptdir = "/home/wanja/DFT--/PlaineDFT/example/debug_nonloc/"
pushfirst!(PyVector(pyimport("sys")."path"), scriptdir)
py_main = pyimport("test_nonloc")

push!(LOAD_PATH, scriptdir)
include(scriptdir*"test_nonloc.jl")

test = "LiH"

py = py_main.main(test)
py_Atoms = py[1]
py_E = py[2]
py_V = py[3]
py_W = py[4]
py_bp = py[5]

jl = main(test)
jl_Ham = jl[1]
jl_E = jl[2]
jl_V = jl[3]
jl_W = jl[4]
jl_bp = jl[5]

# Sorted, cut-off squared magnitudes of G-vectors
println("G2c: ", isapprox(py_Atoms.G2c, jl_Ham.pw.gvec.G2[jl_Ham.pw.gvecw.idx_gw2g[1]]; rtol=0))
# Sorted G-vectors (mine have to be transposed)
println("Gc: ", isapprox(py_Atoms.Gc', jl_Ham.pw.gvec.G[:,jl_Ham.pw.gvecw.idx_gw2g[1]]; rtol=0))
# Non-local pseudopotential parameters
println("NbetaNL: ", isapprox(py_Atoms.NbetaNL, jl_Ham.pspotNL.NbetaNL; rtol=0))
println("prj2beta: ", isapprox(py_Atoms.prj2beta, jl_Ham.pspotNL.prj2beta; rtol=0))
println("betaNL: ", isapprox(py_Atoms.betaNL, jl_Ham.pspotNL.betaNL[1]; rtol=1.0e-12))
# Applied wave function (Psi' * betaNL)*
println("betaNL_psi: ", isapprox(py_bp, jl_bp; rtol=1.0e-12))
println("Psi: ", isapprox(py_W, jl_W; rtol=0))
println("Vnl: ", isapprox(py_V, jl_V[1]; rtol=1.0e-12))
println("Enl: ", isapprox(py_E, jl_E; rtol=1.0e-12))
