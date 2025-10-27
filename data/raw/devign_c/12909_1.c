grlib_apbuart_writel(void *opaque, target_phys_addr_t addr, uint32_t value)

{

    UART          *uart = opaque;

    unsigned char  c    = 0;



    addr &= 0xff;



    /* Unit registers */

    switch (addr) {

    case DATA_OFFSET:

        c = value & 0xFF;

        qemu_chr_write(uart->chr, &c, 1);

        return;



    case STATUS_OFFSET:

        /* Read Only */

        return;



    case CONTROL_OFFSET:

        /* Not supported */

        return;



    case SCALER_OFFSET:

        /* Not supported */

        return;



    default:

        break;

    }



    trace_grlib_apbuart_unknown_register("write", addr);

}
