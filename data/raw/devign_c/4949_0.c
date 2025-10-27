static void omap_mpui_io_write(void *opaque, target_phys_addr_t addr,

                               uint64_t value, unsigned size)

{

    /* FIXME: infinite loop */

    omap_badwidth_write16(opaque, addr, value);

}
