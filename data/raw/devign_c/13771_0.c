static uint64_t omap_dpll_read(void *opaque, target_phys_addr_t addr,

                               unsigned size)

{

    struct dpll_ctl_s *s = (struct dpll_ctl_s *) opaque;



    if (size != 2) {

        return omap_badwidth_read16(opaque, addr);

    }



    if (addr == 0x00)	/* CTL_REG */

        return s->mode;



    OMAP_BAD_REG(addr);

    return 0;

}
