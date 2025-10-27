static uint64_t omap_clkdsp_read(void *opaque, target_phys_addr_t addr,

                                 unsigned size)

{

    struct omap_mpu_state_s *s = (struct omap_mpu_state_s *) opaque;



    if (size != 2) {

        return omap_badwidth_read16(opaque, addr);

    }



    switch (addr) {

    case 0x04:	/* DSP_IDLECT1 */

        return s->clkm.dsp_idlect1;



    case 0x08:	/* DSP_IDLECT2 */

        return s->clkm.dsp_idlect2;



    case 0x14:	/* DSP_RSTCT2 */

        return s->clkm.dsp_rstct2;



    case 0x18:	/* DSP_SYSST */

        return (s->clkm.clocking_scheme << 11) | s->clkm.cold_start |

                (s->cpu->env.halted << 6);      /* Quite useless... */

    }



    OMAP_BAD_REG(addr);

    return 0;

}
