static bool fw_cfg_comb_valid(void *opaque, target_phys_addr_t addr,

                                  unsigned size, bool is_write)

{

    return (size == 1) || (is_write && size == 2);

}
