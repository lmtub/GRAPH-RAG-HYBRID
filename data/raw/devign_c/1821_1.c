static inline int compare_masked(uint64_t x, uint64_t y, uint64_t mask)

{

    return (x & mask) == (y & mask);

}
