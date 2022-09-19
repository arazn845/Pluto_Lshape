### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ b04a6435-29fa-433d-88b4-2df59c84d043
using GLPK, JuMP

# ╔═╡ 178b79d7-0378-4620-8615-9c517b6f80f7
using Printf

# ╔═╡ 3039bb41-4657-45fc-bf3a-a75da3e74dbc
using PlutoUI

# ╔═╡ 1c0e944e-37b1-11ed-2d98-317bb205a4e3
md"""
# Benders decomposition julia
### with linear sub problem
"""

# ╔═╡ e7f98549-da6e-4d5e-acac-dfe953e38e31
md"""

### Problem

$$\begin{aligned}
\text{min}        \ & c_1^\top x+c_2^\top y   \\
\text{subject to} \ & A_1 x+ A_2 y \le b      \\
                    & x \ge 0                 \\
                    & y \ge 0                 \\
                    & x \in \mathbb{Z}^n
\end{aligned}$$
"""

# ╔═╡ 0721d470-17e8-4751-b419-0c9c28b28864
md"""

### Main problem

$$\begin{aligned}
\text{min}        \ & c_1^\top x + θ \\
\text{subject to} \ & x \ge 0             \\
                    & x \in \mathbb{Z}^n \\
                    & θ ≥ θ^k + -λ_k^\top A_1 (x - x_k).
\end{aligned}$$
"""

# ╔═╡ de8fd381-e12a-4c8e-8e24-4f7728d244ca
md"""

### Sub problem

$$\begin{aligned}
θ =      & \text{min}          \ & c_2^\top y                        \\
         & \text{subject to}   \ & A_2 y \le b - A_1 x & \quad [λ] \\
         &                       & y \ge 0,
\end{aligned}$$
"""

# ╔═╡ dc2a5924-f3da-4416-9c91-b2eca46fe593
md"""
# Example

$$\min x_1 + 4 x_2 + 2 y_1 + 3 y_2$$

$$x_1 -3 x_2 + y_1 -2 y_2 ≤ -2$$

$$x_1 -2 x_2 -1 y_1 -1 y_2 ≤ -3$$

$$x , y ≥ 0$$

"""

# ╔═╡ c6639662-867c-4980-945b-c0ed2a3fe406
md"""
### Main problem

$$\min x_1 + 4 x_2 + θ$$

$$x ≥ 0$$

"""

# ╔═╡ 77c96d23-d93d-4e27-a72f-f4288b5e2ccd
md"""
### Sub problem

$$\min 2 y_1 + 3 y_2$$

$$x_1 -3 x_2 + y_1 -2 y_2 ≤ -2$$

$$x_1 -2 x_2 -1 y_1 -1 y_2 ≤ -3$$

$$x , y ≥ 0$$

"""

# ╔═╡ 0db4199e-838c-41ab-a0a5-d8daf6d5c250
md"""
### Parameters
"""

# ╔═╡ a5645905-cc83-4e22-ac18-0cd08d897f04
begin
	c1 = [1, 4]
	c2 = [2, 3]
	b2 = [-2; -3]
	A1 = [1 -3; -1 -3]
	A2 = [1 -2; -1 -1]
	M = -1000;
end

# ╔═╡ 9d080a6f-cf2c-4683-9e12-d13890d582de
md"""
### function for main problem
"""

# ╔═╡ 2b65ade6-66ec-4b30-b4b5-d8ca737461c9
md"""

### Note: do not put main problem inside a function

"""

# ╔═╡ 63c928f3-910f-4198-96cb-39e99d9eff43
begin
	main = Model(GLPK.Optimizer)
	@variable(main, 0 ≤ x[1:2])
	@variable(main, -1000 ≤ θ)
	@objective(main, Min, c1' * x + θ)
end

# ╔═╡ 491e5ea6-413a-42ac-80b4-7f6cc96870d0
md"""
### function for sub problem
"""

# ╔═╡ cdbe4eab-737f-4eac-a75a-9554a0f9fe18
md"""

These are what I need to get from sub problem

-  $o$ objective function

-  $y$ variable value

-  $λ$ dual (simplex multiplier)
"""

# ╔═╡ a5404816-cfc0-4dcc-94ca-f47729fa52dd
function sub(x)
	sub = Model(GLPK.Optimizer)
	@variable(sub, 0 ≤ y[1:2])
	@objective(sub, Min, c2' * y)
	@constraint(sub, A1 * x + A2 * y .≤ b2)
	#####################
	optimize!(sub)
	#####################
	o = objective_value(sub)
	y = value.(y)
	all_cons = all_constraints(sub, AffExpr, MOI.LessThan{Float64})
	λ = dual.(all_cons)
	return Dict('o' => o , 'y' => y , 'λ' => λ)
end

# ╔═╡ 70f91253-8113-4a76-b5c8-60f94c92921d
sub([0 ,0])

# ╔═╡ b756b386-dbff-4dd8-882f-2dc21931dac1
sub([0 ,0])['λ']

# ╔═╡ 49d52fcc-9b9a-4843-ba53-9cddcc25baba
sub([0 ,0])['o']

# ╔═╡ d811a092-f2cb-48d0-b70a-524529ae4237
function print_iteration(k, args...)
    f(x) = Printf.@sprintf("%12.4e", x)
    println(lpad(k, 9), " ", join(f.(args), " "))
    return
end

# ╔═╡ 84ab78f6-c7af-4ef2-989a-c2283023e911
md"""

-  $c_1 x + c_2 y$

-  main problem $c_1 xᵏ + θ$ lower bound

-  $c_1 xᵏ + c_2 yᵏ$ upper bound

-  define an **optimality gap**

-  compare the **optimality gap** with the **current gap**



"""

# ╔═╡ d54ed2a0-c19f-4365-bc49-71d875e494fc
optimality_gap = 1e-6

# ╔═╡ 19857405-cf59-4abe-ba58-de788117a041
md"""
$$θ ≥ θᵏ - λᵏ A1 ( x - xᵏ )$$
"""

# ╔═╡ a5027109-414d-47c8-bf98-54d5127bc044
md"
### other stuff
"

# ╔═╡ 7eb4a3a4-d39f-4831-9d7b-d07500dc12f9
objective_value(main)

# ╔═╡ f9aad74d-ee45-493c-93da-5668198688ea
all_constraints(main, AffExpr, MOI.GreaterThan{Float64})

# ╔═╡ 6eeec4ab-825e-47be-9f99-dd557ee6f2b8
optimize!(main)

# ╔═╡ c37867bc-71ff-4512-867f-34bbdbbd6e64
lb = objective_value(main)

# ╔═╡ b681012e-3c9d-440e-9241-3616df25eca0
xᵏ = value.(x)

# ╔═╡ 039c63e6-e049-43bf-8ef7-5cbd1724aed5
ub = c1' * xᵏ + c2' * sub(xᵏ)['y']

# ╔═╡ 63583ca3-b21f-4489-bdcc-ed5bee27cdb8
function gap(lb , up)
	gap = (ub - lb) / ub
	return gap
end

# ╔═╡ 0d9f2da0-2d4b-4837-96ed-65bdf39fa669
for k in 1:10
	optimize!(main)
	lb = objective_value(main)
	xᵏ = value.(x)
	ub = c1' * xᵏ + c2' * sub(xᵏ)['y']
	gap = gap(lb , ub)
	print_iteration(k, lb, ub, gap)
	if gap <optimality_gap
		println("***** congrats we are at optimality ******")
		break
	end
	cut = @constraint(main, θ ≥ sub(xᵏ)['o'] + (sub(xᵏ)['λ'])' * A1 * (x .- xᵏ) )
	@info "we are adding the cut $(cut)"
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
GLPK = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[compat]
GLPK = "~1.1.0"
JuMP = "~1.3.0"
PlutoUI = "~0.7.40"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

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

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

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

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

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

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

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

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

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

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "4e675d6e9ec02061800d6cfb695812becbd03cdf"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.4"

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

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "a602d7b0babfca89005da04d89223b867b55319f"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.40"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

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
git-tree-sha1 = "efa8acd030667776248eabb054b1836ac81d92f0"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.7"

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

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

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
# ╟─1c0e944e-37b1-11ed-2d98-317bb205a4e3
# ╠═b04a6435-29fa-433d-88b4-2df59c84d043
# ╠═178b79d7-0378-4620-8615-9c517b6f80f7
# ╠═3039bb41-4657-45fc-bf3a-a75da3e74dbc
# ╟─e7f98549-da6e-4d5e-acac-dfe953e38e31
# ╟─0721d470-17e8-4751-b419-0c9c28b28864
# ╟─de8fd381-e12a-4c8e-8e24-4f7728d244ca
# ╟─dc2a5924-f3da-4416-9c91-b2eca46fe593
# ╟─c6639662-867c-4980-945b-c0ed2a3fe406
# ╟─77c96d23-d93d-4e27-a72f-f4288b5e2ccd
# ╟─0db4199e-838c-41ab-a0a5-d8daf6d5c250
# ╠═a5645905-cc83-4e22-ac18-0cd08d897f04
# ╟─9d080a6f-cf2c-4683-9e12-d13890d582de
# ╟─2b65ade6-66ec-4b30-b4b5-d8ca737461c9
# ╠═63c928f3-910f-4198-96cb-39e99d9eff43
# ╟─491e5ea6-413a-42ac-80b4-7f6cc96870d0
# ╟─cdbe4eab-737f-4eac-a75a-9554a0f9fe18
# ╠═a5404816-cfc0-4dcc-94ca-f47729fa52dd
# ╠═70f91253-8113-4a76-b5c8-60f94c92921d
# ╠═b756b386-dbff-4dd8-882f-2dc21931dac1
# ╠═49d52fcc-9b9a-4843-ba53-9cddcc25baba
# ╠═d811a092-f2cb-48d0-b70a-524529ae4237
# ╟─84ab78f6-c7af-4ef2-989a-c2283023e911
# ╠═d54ed2a0-c19f-4365-bc49-71d875e494fc
# ╠═63583ca3-b21f-4489-bdcc-ed5bee27cdb8
# ╟─19857405-cf59-4abe-ba58-de788117a041
# ╠═0d9f2da0-2d4b-4837-96ed-65bdf39fa669
# ╠═a5027109-414d-47c8-bf98-54d5127bc044
# ╠═7eb4a3a4-d39f-4831-9d7b-d07500dc12f9
# ╠═f9aad74d-ee45-493c-93da-5668198688ea
# ╠═6eeec4ab-825e-47be-9f99-dd557ee6f2b8
# ╠═c37867bc-71ff-4512-867f-34bbdbbd6e64
# ╠═b681012e-3c9d-440e-9241-3616df25eca0
# ╠═039c63e6-e049-43bf-8ef7-5cbd1724aed5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
