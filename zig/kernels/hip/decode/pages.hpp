#pragma once

// Paged KV: a layer's keys (or values) live in a pool of `count` pages a head, each page 64 positions of d values, and a stream
// names its pages in a table (page ids, one per 64 positions). A head's pages sit side by side, so a stream whose pages were
// taken in order reads a head's keys as one stretch. Every kernel that reads or writes a cache goes through it.
constexpr int kPageTokens = 64;

// The element offset in a pool of `count` pages a head of head `h`, slot `slot` of page `page`.
__device__ __forceinline__ long long page_row(unsigned page, unsigned count, int h, int slot, int d) {
    return ((static_cast<long long>(h) * count + page) * kPageTokens + slot) * d;
}

// The element offset in a pool of head `h` at stream position `pos`.
__device__ __forceinline__ long long page_at(const unsigned* table, unsigned count, int h, int pos, int d) {
    return page_row(table[pos / kPageTokens], count, h, pos % kPageTokens, d);
}
