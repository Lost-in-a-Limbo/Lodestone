#pragma once

// Phase 6 — attribute store, predicate evaluation, and the three filtering
// strategies. This is the phase the project exists for.
//
// Placeholder. What lands here:
//
//   AttributeStore    per-vector metadata (categorical tags + one numeric field)
//   Predicate         evaluated against a roaring-bitmap-style membership set
//   strategy A        pre-filter  — materialise the passing set, brute force it
//   strategy B        post-filter — search with over-fetch factor f, discard misses
//   strategy C        in-filter   — predicate check before distance during
//                                  traversal, with two-hop expansion when
//                                  one-hop candidates are exhausted (ACORN-1)
//
// Deliberately empty until Phase 3 exists. A filter strategy written before
// there is a graph to filter is a guess.

#include "lodestone/types.hpp"

namespace lodestone {

class AttributeStore;

} // namespace lodestone
