static uint64_t fw_cfg_data_mem_read(void *opaque, hwaddr addr,

                                     unsigned size)

{

    return fw_cfg_read(opaque);

}
