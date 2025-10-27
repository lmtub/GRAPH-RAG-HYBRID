static void amdvi_mmio_trace(hwaddr addr, unsigned size)

{

    uint8_t index = (addr & ~0x2000) / 8;



    if ((addr & 0x2000)) {

        /* high table */

        index = index >= AMDVI_MMIO_REGS_HIGH ? AMDVI_MMIO_REGS_HIGH : index;

        trace_amdvi_mmio_read(amdvi_mmio_high[index], addr, size, addr & ~0x07);

    } else {

        index = index >= AMDVI_MMIO_REGS_LOW ? AMDVI_MMIO_REGS_LOW : index;

        trace_amdvi_mmio_read(amdvi_mmio_high[index], addr, size, addr & ~0x07);

    }

}
