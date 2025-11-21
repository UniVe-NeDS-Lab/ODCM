# MIT License

# Copyright (c) [2022] [Gabriele Gemmi gabriele.gemmi@unive.it]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

using Genie, Genie.Requests, Genie.Renderer.Json
using Graphs, MetaGraphs, GraphIO
using EzXML
include("forbidden.jl")


route("/espfp", method=POST) do
    #@show jsonpayload()
    data = jsonpayload()
    g = load_graph(data["n"], data["edgelist"])
    paths = solvePricing(g, data["gws"])
    json(paths)
end

# Start the app!
up(8888, host="0.0.0.0", verbose=true)



function load_graph(node_number, edgelist)
    G = MetaGraph(node_number)
    for l in split(edgelist, '\n')
        if length(l) ≤ 0
            continue
        end
        l = split(l, " ")
        src = parse(Int64, l[1])
        dst = parse(Int64, l[2])
        dist = parse(Float32, l[3])
        add_edge!(G, src, dst)
        set_prop!(G, src, dst, :dist, dist)
    end
    return G
end


