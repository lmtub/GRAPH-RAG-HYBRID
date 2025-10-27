static bool sys_ops_accepts(void *opaque, target_phys_addr_t addr,

                            unsigned size, bool is_write)

{

    return is_write && size == 4;

}
