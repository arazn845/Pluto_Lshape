### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 13c2b0de-994a-4f7e-9f6e-9f4fda82942f
using JuMP, GLPK, NLPModels, NLPModelsJuMP

# ╔═╡ eed46771-211f-4f40-b02e-7fbf87fd6d00
using DataFrames

# ╔═╡ 8d1ecb81-b292-47f9-84d7-eeb488d901c2
md"# Optimality cuts"

# ╔═╡ 870d007a-f2a0-4313-a1eb-d7af267b4b8a
md"""
 

### general type 

$$θ ≥ e_l - E_l x$$ 

"""

# ╔═╡ 532c3d40-3746-11ed-313b-5da6de95256e
md"""
$$e = \sum_{i=1}^{K} p_i λ_i h_i$$
"""

# ╔═╡ d361e47d-5a20-4608-bf43-14d867aa5ee8
md"""
$$E = \sum_{i=1}^{K} p_i λ_i T_i$$
"""

# ╔═╡ 7d4348bc-b219-4702-ad4d-f9be949b42c1
md"""

-  $p$ comes from problem description

-  $λ$ comes from solving the sub problem

-  $h$ and $T$ are coming from sub problem with no solution

"""

# ╔═╡ 938860d7-0337-46f1-9221-00144c13c72f


# ╔═╡ c99b15e6-2e2f-4579-9f08-853d0da77dae
md"""
# Example
"""

# ╔═╡ 1cb458c7-7d82-44d9-900d-d2f8bd676b10
md"""
### Stage 1

$$\min 100 x_1 + 150 x_2$$
$$x_1 + x_2 ≤ 120$$
$$- x_1 ≤ -40$$
$$- x_2 ≤ -20$$

"""

# ╔═╡ 9d75f185-7154-4793-8b67-47b32bf0ef83
begin
	c1 = [100 ; 150]
	A1 = [1 1 ; -1 0 ; 0 -1]
	b1 = [120 ; -40 ; -20]
	main = Model(GLPK.Optimizer)
	@variable(main, x[1:2])
	@variable(main, -100000 ≤ θ)
	@objective(main, Min, c1' * x + θ)
	@constraint(main, A1 * x .≤ b1)
	optimize!(main)
	x = value.(x)
	θ = value(θ)
end

# ╔═╡ fdbb157e-8740-4573-ab06-57a12cb4b4c2
x

# ╔═╡ 92015914-c62f-420b-9752-8ca9995f81f7
θ

# ╔═╡ cbfd7bd8-e70e-43f9-afc7-425e8e68c136
md"""
### sub problem 1

$$\min -24 y_1 - 28 y_2$$
$$6 y_1 + 10 y_2 -60 x_1 ≤ 0$$
$$8 y_1 + 5 y_2 -80 x_2 ≤ 0$$
$$y_1 ≤ 500$$
$$y_2 ≤ 100$$

### sub problem 2

$$\min -28 y_1 - 32 y_2$$
$$6 y_1 + 10 y_2 -60 x_1 ≤ 0$$
$$8 y_1 + 5 y_2 -80 x_2 ≤ 0$$
$$y_1 ≤ 300$$
$$y_2 ≤ 300$$

"""

# ╔═╡ 3619bb31-38bb-4edb-9838-37d8e701ff8a
begin
	I = 2
	c2 = [-24 -28 ; -28 -32];
	A2 = [6 10 ; 8 5 ; 1 0 ; 0 1];
	A3 = [-60 0 ; 0 -80 ; 0 0 ; 0 0];
	b2 = [0 0 500 100 ; 0 0 300 300];
end

# ╔═╡ fed35910-8056-4bb2-9f1e-225f58a9925b
solve_sub = function solve_sub()
	λ = zeros(2,4)
	for i in 1:I
		sub = Model(GLPK.Optimizer)
		@variable(sub, 0 ≤ y[1:2])
		@objective(sub, Min, c2[i , :]' * y)
		@constraint(sub, A2 * y + A3 * x .≤ b2[i , :])
		optimize!(sub)
		all_cons = all_constraints(sub, AffExpr, MOI.LessThan{Float64})
		λ[i , :] = dual.(all_cons) 
	end
	return λ
end

# ╔═╡ 300b5185-a487-453e-a3d0-136247430f8c
λ = solve_sub()

# ╔═╡ be03f53c-09cf-4021-8418-cef8d0e4f326
no_solve_sub = function no_solve_sub()
	T = zeros(2,4)
	h = zeros(2,4)
	for i in 1:I
		sub = Model(GLPK.Optimizer)
		@variable(sub, x[1:2])
		@variable(sub, 0 ≤ y[1:2])
		@objective(sub, Min, c2[i , :]' * y)
		@constraint(sub, A2 * y + A3 * x .≤ b2[i , :])
		nlp = MathOptNLPModel(sub)
		l = zeros(nlp.meta.nvar)
		h[i , :] = nlp.meta.ucon 
		#all_var = all_variables(sub)
		#var_index = [all_var[s].index.value for s in 1:length(all_var)]
		#df = DataFrame(varName = all_var , varIndex = var_index) 
		T = Matrix(jac(nlp,l))[ : , 1:2]
	end
	return (T, h)
end

# ╔═╡ 8d4f5987-ee3e-48c1-94f6-bcd0ab603356
T = no_solve_sub()[1]

# ╔═╡ b2272be4-f71b-446a-b574-13cc333175b4
h = no_solve_sub()[2]

# ╔═╡ 2d41ef5f-4e34-480c-8bea-48af7b6a2b83
no_solve_sub()[3]

# ╔═╡ 36fa09e6-132f-4779-b0c4-97413fea65ac
md"""
$$e = \sum_{i=1}^{K} p_i λ_i h_i$$
"""

# ╔═╡ 38b8e97d-39d5-4f2a-b50f-08e1d4b4c175
p = [0.4 , 0.6]

# ╔═╡ 075fa0cd-a785-493b-94f6-43a2f06edec5
λ

# ╔═╡ 44e29254-6bcf-4786-bc4d-a76fac01a28d
h

# ╔═╡ ff431dcc-ff2a-49ed-802a-6b12576d9a20
e = sum(p[i] * λ[i,:]' * h[i,:] for i in 1:I)

# ╔═╡ 309d8e7f-6e70-45d5-9525-d527d1ddca55
md"""
$$E = \sum_{i=1}^{K} p_i λ_i T_i$$
"""

# ╔═╡ 447bb9e3-32a1-45dd-9511-55c0e10e4330
T

# ╔═╡ bc3a5be2-ddc4-4be0-adbd-e652c4540d92
E = sum(p[i] * λ[i,:]' * T for i in 1:I)

# ╔═╡ 6a428c04-6345-4713-9927-3e12afa15ac5
md"""
- let $w = e - E x^v$ 

- if $θ ≥ w$ stop otherwise add the constraint
"""

# ╔═╡ f357fb71-cdf4-461b-aedf-8fb32eecd832
w = e - E * x

# ╔═╡ b8672a76-1212-469b-ae64-3e37e5417bc2
θ

# ╔═╡ 53126dd6-1617-4251-800f-78f7440b69a9
if θ > w 
	println("the algorithm is finished with optimality")
else
	println("adding the cut θ ≥ $(e) - $(E[1])x_1 - $(E[2])x_2")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLPK = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
NLPModels = "a4795742-8479-5a88-8948-cc11e1c8c1a6"
NLPModelsJuMP = "792afdf1-32c1-5681-94e0-d7bf7a5df49e"

[compat]
DataFrames = "~1.3.5"
GLPK = "~1.1.0"
JuMP = "~1.3.0"
NLPModels = "~0.19.1"
NLPModelsJuMP = "~0.12.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AMD]]
deps = ["Libdl", "LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "00163dc02b882ca5ec032400b919e5f5011dbd31"
uuid = "14f7f29c-3bd6-536c-9a0b-7339e30b5a3e"
version = "0.5.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "dc4405cee4b2fe9e1108caec2d760b7ea758eca2"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.5"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5856d3031cdb1f3b2b6340dfdc66b6d9a149a374"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.2.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "6bce52b2060598d8caaed807ec6d6da2a1de949e"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.5"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "992a23afdb109d0d2f8802a30cf5ae4b1fe7ea68"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.1"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLPK]]
deps = ["GLPK_jll", "MathOptInterface"]
git-tree-sha1 = "e357b935632e89a02cf7f8f13b4f3f59cef479c8"
uuid = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
version = "1.1.0"

[[GLPK_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "fe68622f32828aa92275895fdb324a85894a5b1b"
uuid = "e8aa6df9-e6ca-548a-97ff-1f85fc5b8b98"
version = "5.0.1+0"

[[GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[JuMP]]
deps = ["LinearAlgebra", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "Printf", "SparseArrays"]
git-tree-sha1 = "906e2325c22ba8aaed432677d0a8d5cf24c9ea9e"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.3.0"

[[LDLFactorizations]]
deps = ["AMD", "LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "743544bcdba7b4ad744bfd5d062c977a9df553a7"
uuid = "40e66cde-538c-5869-a4ad-c39174c6795b"
version = "0.9.0"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LinearOperators]]
deps = ["FastClosures", "LDLFactorizations", "LinearAlgebra", "Printf", "SparseArrays", "TimerOutputs"]
git-tree-sha1 = "efbe1aab85805bf654e209e9c586aaa7ec695edf"
uuid = "5c8ed15e-5a4c-59e4-a42b-c7e8811fb125"
version = "2.4.0"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "81a98bed15dc1d49ed6afa2dc65272e402a1704b"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.8.1"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "4e675d6e9ec02061800d6cfb695812becbd03cdf"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.4"

[[NLPModels]]
deps = ["FastClosures", "LinearAlgebra", "LinearOperators", "Printf", "SparseArrays"]
git-tree-sha1 = "25ea665d2dacca4786f17a0d34836f95fb1a00cd"
uuid = "a4795742-8479-5a88-8948-cc11e1c8c1a6"
version = "0.19.1"

[[NLPModelsJuMP]]
deps = ["JuMP", "LinearAlgebra", "MathOptInterface", "NLPModels", "Printf", "SparseArrays"]
git-tree-sha1 = "c5f6ecf92a5b21858c156c6ec7c62861bb26ab9f"
uuid = "792afdf1-32c1-5681-94e0-d7bf7a5df49e"
version = "0.12.0"

[[NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3d5bf43e3e8b412656404ed9466f1dcbf7c50269"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "dfec37b90740e3b9aa5dc2613892a3fc155c3b42"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.6"

[[StaticArraysCore]]
git-tree-sha1 = "ec2bd695e905a3c755b33026954b119ea17f2d22"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.3.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "7149a60b01bf58787a1b83dad93f90d4b9afbe5d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.8.1"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "9dfcb767e17b0849d6aaf85997c98a5aea292513"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.21"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─8d1ecb81-b292-47f9-84d7-eeb488d901c2
# ╠═13c2b0de-994a-4f7e-9f6e-9f4fda82942f
# ╠═eed46771-211f-4f40-b02e-7fbf87fd6d00
# ╟─870d007a-f2a0-4313-a1eb-d7af267b4b8a
# ╟─532c3d40-3746-11ed-313b-5da6de95256e
# ╟─d361e47d-5a20-4608-bf43-14d867aa5ee8
# ╟─7d4348bc-b219-4702-ad4d-f9be949b42c1
# ╠═938860d7-0337-46f1-9221-00144c13c72f
# ╟─c99b15e6-2e2f-4579-9f08-853d0da77dae
# ╟─1cb458c7-7d82-44d9-900d-d2f8bd676b10
# ╠═9d75f185-7154-4793-8b67-47b32bf0ef83
# ╠═fdbb157e-8740-4573-ab06-57a12cb4b4c2
# ╠═92015914-c62f-420b-9752-8ca9995f81f7
# ╟─cbfd7bd8-e70e-43f9-afc7-425e8e68c136
# ╠═3619bb31-38bb-4edb-9838-37d8e701ff8a
# ╠═fed35910-8056-4bb2-9f1e-225f58a9925b
# ╠═300b5185-a487-453e-a3d0-136247430f8c
# ╠═be03f53c-09cf-4021-8418-cef8d0e4f326
# ╠═8d4f5987-ee3e-48c1-94f6-bcd0ab603356
# ╠═b2272be4-f71b-446a-b574-13cc333175b4
# ╠═2d41ef5f-4e34-480c-8bea-48af7b6a2b83
# ╟─36fa09e6-132f-4779-b0c4-97413fea65ac
# ╠═38b8e97d-39d5-4f2a-b50f-08e1d4b4c175
# ╠═075fa0cd-a785-493b-94f6-43a2f06edec5
# ╠═44e29254-6bcf-4786-bc4d-a76fac01a28d
# ╠═ff431dcc-ff2a-49ed-802a-6b12576d9a20
# ╠═309d8e7f-6e70-45d5-9525-d527d1ddca55
# ╠═447bb9e3-32a1-45dd-9511-55c0e10e4330
# ╠═bc3a5be2-ddc4-4be0-adbd-e652c4540d92
# ╟─6a428c04-6345-4713-9927-3e12afa15ac5
# ╠═f357fb71-cdf4-461b-aedf-8fb32eecd832
# ╠═b8672a76-1212-469b-ae64-3e37e5417bc2
# ╠═53126dd6-1617-4251-800f-78f7440b69a9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
