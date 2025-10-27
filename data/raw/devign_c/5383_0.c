softusb_write(void *opaque, target_phys_addr_t addr, uint64_t value,

              unsigned size)

{

    MilkymistSoftUsbState *s = opaque;



    trace_milkymist_softusb_memory_write(addr, value);



    addr >>= 2;

    switch (addr) {

    case R_CTRL:

        s->regs[addr] = value;

        break;



    default:

        error_report("milkymist_softusb: write access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }

}
