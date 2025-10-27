static uint64_t boston_platreg_read(void *opaque, hwaddr addr,

                                    unsigned size)

{

    BostonState *s = opaque;

    uint32_t gic_freq, val;



    if (size != 4) {

        qemu_log_mask(LOG_UNIMP, "%uB platform register read", size);

        return 0;

    }



    switch (addr & 0xffff) {

    case PLAT_FPGA_BUILD:

    case PLAT_CORE_CL:

    case PLAT_WRAPPER_CL:

        return 0;

    case PLAT_DDR3_STATUS:

        return PLAT_DDR3_STATUS_LOCKED | PLAT_DDR3_STATUS_CALIBRATED;

    case PLAT_MMCM_DIV:

        gic_freq = mips_gictimer_get_freq(s->cps->gic.gic_timer) / 1000000;

        val = gic_freq << PLAT_MMCM_DIV_INPUT_SHIFT;

        val |= 1 << PLAT_MMCM_DIV_MUL_SHIFT;

        val |= 1 << PLAT_MMCM_DIV_CLK0DIV_SHIFT;

        val |= 1 << PLAT_MMCM_DIV_CLK1DIV_SHIFT;

        return val;

    case PLAT_BUILD_CFG:

        val = PLAT_BUILD_CFG_PCIE0_EN;

        val |= PLAT_BUILD_CFG_PCIE1_EN;

        val |= PLAT_BUILD_CFG_PCIE2_EN;

        return val;

    case PLAT_DDR_CFG:

        val = s->mach->ram_size / G_BYTE;

        assert(!(val & ~PLAT_DDR_CFG_SIZE));

        val |= PLAT_DDR_CFG_MHZ;

        return val;

    default:

        qemu_log_mask(LOG_UNIMP, "Read platform register 0x%" HWADDR_PRIx,

                      addr & 0xffff);

        return 0;

    }

}
