static inline uint64_t vtd_iova_limit(VTDContextEntry *ce)

{

    uint32_t ce_agaw = vtd_ce_get_agaw(ce);

    return 1ULL << MIN(ce_agaw, VTD_MGAW);

}
