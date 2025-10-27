static void fw_cfg_ctl_mem_write(void *opaque, target_phys_addr_t addr,

                                 uint64_t value, unsigned size)

{

    fw_cfg_select(opaque, (uint16_t)value);

}
