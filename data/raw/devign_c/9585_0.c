static uint32_t intel_hda_mmio_readw(void *opaque, target_phys_addr_t addr)

{

    IntelHDAState *d = opaque;

    const IntelHDAReg *reg = intel_hda_reg_find(d, addr);



    return intel_hda_reg_read(d, reg, 0xffff);

}
