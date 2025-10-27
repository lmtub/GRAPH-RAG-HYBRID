static uint64_t softusb_read(void *opaque, target_phys_addr_t addr,

                             unsigned size)

{

    MilkymistSoftUsbState *s = opaque;

    uint32_t r = 0;



    addr >>= 2;

    switch (addr) {

    case R_CTRL:

        r = s->regs[addr];

        break;



    default:

        error_report("milkymist_softusb: read access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }



    trace_milkymist_softusb_memory_read(addr << 2, r);



    return r;

}
