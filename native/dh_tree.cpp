#include "dh_tree.h"

#include <algorithm>
#include <cmath>

void dh_tree_build(const float * lat, int n_block, int top_k, int budget, int max_leaves, int max_depth, dh_tree & t) {
    t.parent.clear();
    t.depth.clear();
    t.tok.clear();
    const size_t w = (size_t) top_k * (1 + top_k);
    max_depth = std::min(max_depth, n_block - 1);
    if (budget <= 0 || max_depth <= 0 || max_leaves <= 0) {
        return;
    }

    struct cand { double p; int32_t d, k, par; bool operator<(const cand & o) const { return p < o.p; } };
    std::vector<cand> heap;
    // node 0 = root: depth 0, predecessor index 0 (the greedy path starts there too)
    std::vector<int32_t> nd_d(1, 0), nd_k(1, 0), n_kids(1, 0);
    int leaves = 1;  // the root counts as one leaf until it gets a child

    auto expand = [&](int32_t node, double p) {
        const int32_t d = nd_d[node] + 1;
        if (d > max_depth) {
            return;
        }
        const float * sc = lat + (size_t) d * w + top_k + (size_t) nd_k[node] * top_k;
        const float mx = *std::max_element(sc, sc + top_k);
        double sum = 0.0;
        for (int k = 0; k < top_k; ++k) {
            sum += std::exp((double) sc[k] - mx);
        }
        for (int k = 0; k < top_k; ++k) {
            const double pk = p * std::exp((double) sc[k] - mx) / sum;
            if (pk > 1e-4) {
                heap.push_back({ pk, d, k, node });
                std::push_heap(heap.begin(), heap.end());
            }
        }
    };

    expand(0, 1.0);
    while (!heap.empty() && t.size() < budget) {
        std::pop_heap(heap.begin(), heap.end());
        const cand c = heap.back();
        heap.pop_back();
        const bool new_leaf = n_kids[c.par] > 0;  // a second child of the same parent opens another branch
        if (new_leaf && leaves >= max_leaves) {
            continue;
        }
        leaves += new_leaf;
        n_kids[c.par] += 1;
        const int32_t id = (int32_t) nd_d.size();
        nd_d.push_back(c.d);
        nd_k.push_back(c.k);
        n_kids.push_back(0);
        t.parent.push_back(c.par);
        t.depth.push_back(c.d);
        t.tok.push_back((llama_token) lat[(size_t) c.d * w + c.k]);
        expand(id, c.p);
    }
}
