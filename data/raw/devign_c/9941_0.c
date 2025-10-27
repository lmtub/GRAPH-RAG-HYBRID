static void fw_cfg_comb_write(void *opaque, target_phys_addr_t addr,

                              uint64_t value, unsigned size)

{

    switch (size) {

    case 1:

        fw_cfg_write(opaque, (uint8_t)value);

        break;

    case 2:

        fw_cfg_select(opaque, (uint16_t)value);

        break;

    }

}
