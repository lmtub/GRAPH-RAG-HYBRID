static inline bool vtd_iova_range_check(uint64_t iova, VTDContextEntry *ce)

{

    /*

     * Check if @iova is above 2^X-1, where X is the minimum of MGAW

     * in CAP_REG and AW in context-entry.

     */

    return !(iova & ~(vtd_iova_limit(ce) - 1));

}
