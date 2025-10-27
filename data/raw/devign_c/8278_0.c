static uint64_t cirrus_mmio_read(void *opaque, target_phys_addr_t addr,

                                 unsigned size)

{

    CirrusVGAState *s = opaque;



    if (addr >= 0x100) {

        return cirrus_mmio_blt_read(s, addr - 0x100);

    } else {

        return cirrus_vga_ioport_read(s, addr + 0x3c0);

    }

}
