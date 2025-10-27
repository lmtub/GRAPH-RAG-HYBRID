static void boston_platreg_write(void *opaque, hwaddr addr,

                                 uint64_t val, unsigned size)

{

    if (size != 4) {

        qemu_log_mask(LOG_UNIMP, "%uB platform register write", size);

        return;

    }



    switch (addr & 0xffff) {

    case PLAT_FPGA_BUILD:

    case PLAT_CORE_CL:

    case PLAT_WRAPPER_CL:

    case PLAT_DDR3_STATUS:

    case PLAT_PCIE_STATUS:

    case PLAT_MMCM_DIV:

    case PLAT_BUILD_CFG:

    case PLAT_DDR_CFG:

        /* read only */

        break;

    case PLAT_SOFTRST_CTL:

        if (val & PLAT_SOFTRST_CTL_SYSRESET) {

            qemu_system_reset_request();

        }

        break;

    default:

        qemu_log_mask(LOG_UNIMP, "Write platform register 0x%" HWADDR_PRIx

                      " = 0x%" PRIx64, addr & 0xffff, val);

        break;

    }

}
