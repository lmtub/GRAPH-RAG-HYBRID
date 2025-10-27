static void intel_hda_mmio_writew(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    IntelHDAState *d = opaque;

    const IntelHDAReg *reg = intel_hda_reg_find(d, addr);



    intel_hda_reg_write(d, reg, val, 0xffff);

}
