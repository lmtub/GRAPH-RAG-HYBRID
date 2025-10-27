static void fw_cfg_data_mem_write(void *opaque, hwaddr addr,

                                  uint64_t value, unsigned size)

{

    FWCfgState *s = opaque;

    uint8_t buf[8];

    unsigned i;



    switch (size) {

    case 1:

        buf[0] = value;

        break;

    case 2:

        stw_he_p(buf, value);

        break;

    case 4:

        stl_he_p(buf, value);

        break;

    case 8:

        stq_he_p(buf, value);

        break;

    default:

        abort();

    }

    for (i = 0; i < size; ++i) {

        fw_cfg_write(s, buf[i]);

    }

}
