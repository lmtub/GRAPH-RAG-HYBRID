static inline target_phys_addr_t get_pgaddr (target_phys_addr_t sdr1,

                                             int sdr_sh,

                                             target_phys_addr_t hash,

                                             target_phys_addr_t mask)

{

    return (sdr1 & ((target_ulong)(-1ULL) << sdr_sh)) | (hash & mask);

}
