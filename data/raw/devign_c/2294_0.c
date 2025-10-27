static uint64_t xilinx_spips_read(void *opaque, hwaddr addr,

                                                        unsigned size)

{

    XilinxSPIPS *s = opaque;

    uint32_t mask = ~0;

    uint32_t ret;



    addr >>= 2;

    switch (addr) {

    case R_CONFIG:

        mask = 0x0002FFFF;

        break;

    case R_INTR_STATUS:

        ret = s->regs[addr] & IXR_ALL;

        s->regs[addr] = 0;

        DB_PRINT("addr=" TARGET_FMT_plx " = %x\n", addr * 4, ret);

        return ret;

    case R_INTR_MASK:

        mask = IXR_ALL;

        break;

    case  R_EN:

        mask = 0x1;

        break;

    case R_SLAVE_IDLE_COUNT:

        mask = 0xFF;

        break;

    case R_MOD_ID:

        mask = 0x01FFFFFF;

        break;

    case R_INTR_EN:

    case R_INTR_DIS:

    case R_TX_DATA:

        mask = 0;

        break;

    case R_RX_DATA:

        rx_data_bytes(s, &ret, s->num_txrx_bytes);

        DB_PRINT("addr=" TARGET_FMT_plx " = %x\n", addr * 4, ret);

        xilinx_spips_update_ixr(s);

        return ret;

    }

    DB_PRINT("addr=" TARGET_FMT_plx " = %x\n", addr * 4, s->regs[addr] & mask);

    return s->regs[addr] & mask;



}
