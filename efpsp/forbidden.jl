# MIT License

# Copyright (c) [2022] [Maxime Elkael maxime_elkael@telecom-sudparis.eu]

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


using Graphs, MetaGraphs, DataStructures, PrettyPrint
import Base.parse
using Profile

function ⊕(A, x)
    new_A = deepcopy(A)
    for i in 1:length(A)
        for j in 1:length(x)
            new_A[i][j] += x[j]
        end
    end
    return new_A
end

function filterForbidden(A, forbidden_paths, num_labels)
    new_A = deepcopy(A)
    for i in 1:length(A)
        for k in 1:length(forbidden_paths)
            if A[i][k+num_labels] >= length(forbidden_paths[k]) - 1
                filter!(x -> x != A[i], new_A)
                continue
            end
        end
    end
    return new_A
end

function filterConstraints(A, valueConstraints, num_labels)
    new_A = deepcopy(A)
    for i in 1:length(A)
        for k in 1:length(valueConstraints)
            if A[i][k+num_labels] >= valueConstraints[k]
                filter!(x -> x != A[i], new_A)
                continue
            end
        end
    end
    return new_A
end


function all_less(x, y)
    reduce(&, (x .<= y))
end

function all_equal(x, y)
    reduce(&, (x .== y))
end


function naive_merge(X, Y)
    X_cpy = deepcopy(X)
    Y_cpy = deepcopy(Y)
    for x in X
        for y in Y
            if all_less(x, y) && !all_equal(x, y)
                filter!(e -> e ≠ y, Y_cpy)
            end
        end
    end
    for y in Y
        for x in X
            if all_less(y, x) && !all_equal(y, x)
                filter!(e -> e ≠ x, X_cpy)
            end
        end
    end
    M = vcat(X_cpy, Y_cpy)
    return unique(M)
end

function dominated(elt, X, numObjs)
    for x in X
        if all_less(x[1:numObjs], elt[1:numObjs]) || all_equal(elt[1:numObjs], x[1:numObjs])
            return true
        end
    end
    return false
end

function addForbiddenLabels!(G, forbidden_paths)
    for e in edges(G)
        label = zeros(length(forbidden_paths))
        for path_idx in 1:length(forbidden_paths)
            p = forbidden_paths[path_idx]
            for i in 1:length(p)-1
                if e.src == p[i] && e.dst == p[i+1]
                    label[path_idx] = 1
                end
                if e.dst == p[i] && e.src == p[i+1]
                    label[path_idx] = 1
                end
            end
        end
        set_prop!(G, e, :forbiddenWeights, label)
    end
end

function get_weights(G, n1, n2, labelNames)
    map(x -> get_prop(G, n1, n2, x), labelNames)
end

function labelCorrectingSP(G, source, labelsObjective, labelsConstraints, maxValueConstraints, forbidden_paths, targeet=nothing)
    # value constranits merges forbidden paths constraints and normla constraints
    valueConstraints = []
    for p in forbidden_paths
        push!(valueConstraints, length(p) - 1)
    end
    valueConstraints = vcat(valueConstraints, maxValueConstraints)

    addForbiddenLabels!(G, forbidden_paths)
    D = Dict()
    DM = []
    Labeled = DataStructures.Queue{Int}()
    for i in 1:nv(G)
        D[i] = []
    end
    D[source] = [zeros(length(labelsObjective) + length(forbidden_paths) + length(labelsConstraints))]
    numConsideredDomination = length(labelsObjective) + length(forbidden_paths)
    enqueue!(Labeled, source)
    while length(Labeled) > 0
        u = dequeue!(Labeled)
        for j in outneighbors(G, u)
            #println(j, " ", u)
            weightsObj = get_weights(G, u, j, labelsObjective)
            weightsConstraints = get_weights(G, u, j, labelsConstraints)
            forbiddenLabels = get_prop(G, u, j, :forbiddenWeights)
            A = D[j]
            B = D[u] ⊕ vcat(weightsObj, forbiddenLabels, weightsConstraints)
            cA = deepcopy(A)
            cB = deepcopy(B)
            DM = naive_merge(A, B)
            DM = filterConstraints(DM, valueConstraints, length(labelsObjective))
            if targeet != j && targeet != nothing && D[targeet] != []
                DM_cpy = deepcopy(DM)
                for elt in DM
                    if dominated(elt, D[targeet], length(labelsObjective))
                        filter!(x -> x != elt, DM_cpy)
                    end
                end
                DM = DM_cpy
            end
            if DM != D[j]
                D[j] = DM
                if !(j in Labeled)
                    enqueue!(Labeled, j)
                end
            end
        end
    end
    return D
end


function get_matching_label(labels, G, n, node, l, labelsObjective, labelsConstraints)
    for l2 in labels[n]
        weightsObjective = get_weights(G, n, node, labelsObjective)
        weightsConstraints = get_weights(G, n, node, labelsConstraints)
        forbiddenLabels = get_prop(G, n, node, :forbiddenWeights)
        s = [l2] ⊕ vcat(weightsObjective, forbiddenLabels, weightsConstraints)
        if all_equal(s[1], l)
            return l2
        end
    end
    return nothing
end

function retrievePaths(labels, G, d, labelsObjective, labelsConstraints)
    paths = []
    for l in labels[d]
        path = [d]
        node = d
        while sum(labels[node][1]) != 0
            for n in inneighbors(G, node)
                new_label = get_matching_label(labels, G, n, node, l, labelsObjective, labelsConstraints)
                if new_label != nothing
                    node = n
                    l = new_label
                    push!(path, node)

                    break
                end
            end
        end
        reverse!(path)
        push!(paths, path)
    end
    return paths
end

function addDummyNodes(G, sources, labels)
    G2 = MetaGraph(nv(G) * 2 + 1)
    for e in edges(G)
        properties = deepcopy(props(G, e))
        add_edge!(G2, e.src, e.dst, properties)
    end
    for n in 1:nv(G)
        properties = Dict()
        for l in labels
            properties[l] = 1.0
        end
        add_edge!(G2, n, n + nv(G), properties)
    end
    for n in sources
        properties = Dict()
        for l in labels
            properties[l] = 1.0
        end
        add_edge!(G2, n, nv(G) * 2 + 1, properties)
    end
    G2
end

function solvePricing(G, source)
    dummyG = deepcopy(G)
    labelsObjective = [:dist]
    dummyG = addDummyNodes(G, source, labelsObjective)
    fake_src = nv(dummyG)
    all_paths = []
    for d in 1:nv(G)
        forbidden_paths = []
        for s in source
            push!(forbidden_paths, [fake_src, s, d, d + nv(G)])
        end
        labels = labelCorrectingSP(dummyG, fake_src, labelsObjective, [], [], forbidden_paths)
        paths = retrievePaths(labels, dummyG, d + nv(G), labelsObjective, [])
        push!(all_paths, paths)
    end
    all_paths

end