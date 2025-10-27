static bool vexpress_cfgctrl_write(arm_sysctl_state *s, unsigned int dcc,

                                   unsigned int function, unsigned int site,

                                   unsigned int position, unsigned int device,

                                   uint32_t val)

{

    /* We don't support anything other than DCC 0, board stack position 0

     * or sites other than motherboard/daughterboard:

     */

    if (dcc != 0 || position != 0 ||

        (site != SYS_CFG_SITE_MB && site != SYS_CFG_SITE_DB1)) {

        goto cfgctrl_unimp;

    }



    switch (function) {

    case SYS_CFG_OSC:

        if (site == SYS_CFG_SITE_MB && device < sizeof(s->mb_clock)) {

            /* motherboard clock */

            s->mb_clock[device] = val;

            return true;

        }

        if (site == SYS_CFG_SITE_DB1 && device < s->db_num_clocks) {

            /* daughterboard clock */

            s->db_clock[device] = val;

            return true;

        }

        break;

    case SYS_CFG_MUXFPGA:

        if (site == SYS_CFG_SITE_MB && device == 0) {

            /* Select whether video output comes from motherboard

             * or daughterboard: log and ignore as QEMU doesn't

             * support this.

             */

            qemu_log_mask(LOG_UNIMP, "arm_sysctl: selection of video output "

                          "not supported, ignoring\n");

            return true;

        }

        break;

    case SYS_CFG_SHUTDOWN:

        if (site == SYS_CFG_SITE_MB && device == 0) {

            qemu_system_shutdown_request();

            return true;

        }

        break;

    case SYS_CFG_REBOOT:

        if (site == SYS_CFG_SITE_MB && device == 0) {

            qemu_system_reset_request();

            return true;

        }

        break;

    case SYS_CFG_DVIMODE:

        if (site == SYS_CFG_SITE_MB && device == 0) {

            /* Selecting DVI mode is meaningless for QEMU: we will

             * always display the output correctly according to the

             * pixel height/width programmed into the CLCD controller.

             */

            return true;

        }

    default:

        break;

    }



cfgctrl_unimp:

    qemu_log_mask(LOG_UNIMP,

                  "arm_sysctl: Unimplemented SYS_CFGCTRL write of function "

                  "0x%x DCC 0x%x site 0x%x position 0x%x device 0x%x\n",

                  function, dcc, site, position, device);

    return false;

}
