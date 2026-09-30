// dh_tree: build a draft token tree from a DFlash2 selector lattice (tree verification, see docs/SPEC_STRATEGY.md).
//
// The DFlash2 selector scores, for every draft position d, top_k candidate tokens and a top_k x top_k table
// "candidate k at d-1 -> candidate k' at d". The plain draft is the greedy path through it. Here the verify
// budget (number of draft tokens checked by the target per round) is spent on the most probable paths instead:
// best-first by path probability, so second / third choices at uncertain positions and branches at different
// positions compete with extending the main line. The target samples one token per node and follows the child
// that matches, so the output distribution is unchanged (greedy or sampled).
#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

struct dh_tree {
    // nodes 1..n in expansion order (a parent always comes before its children); parent 0 = the root (id_last)
    std::vector<int32_t>     parent;
    std::vector<int32_t>     depth;   // 1 = first draft position
    std::vector<llama_token> tok;
    int size() const { return (int) tok.size(); }
};

// lat: n_block rows of top_k * (1 + top_k) floats (candidate ids, then scores[pred][k]); row 0 is the anchor.
// budget: max nodes; max_leaves: max branches (one KV sequence each); max_depth: deepest position allowed.
void dh_tree_build(const float * lat, int n_block, int top_k, int budget, int max_leaves, int max_depth, dh_tree & t);
