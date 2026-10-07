#pragma once

// Paged KV: a layer's keys (or values) live in a pool of pages, each kv_heads x 64 positions x d values, and a stream
// names its pages in a table (page ids, one per 64 positions). Every kernel that reads or writes a cache goes through it.
constexpr int kPageTokens = 64;

// The element offset in a pool of head `h`, slot `slot` of page `page`.
__device__ __forceinline__ long long page_row(unsigned page, int kv_heads, int h, int slot, int d) {
    return ((static_cast<long long>(page) * kv_heads + h) * kPageTokens + slot) * d;
}

// The element offset in a pool of head `h` at stream position `pos`.
__device__ __forceinline__ long long page_at(const unsigned* table, int kv_heads, int h, int pos, int d) {
    return page_row(table[pos / kPageTokens], kv_heads, h, pos % kPageTokens, d);
}
