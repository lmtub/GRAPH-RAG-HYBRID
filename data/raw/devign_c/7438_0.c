static uint64_t arm_sysctl_read(void *opaque, target_phys_addr_t offset,

                                unsigned size)

{

    arm_sysctl_state *s = (arm_sysctl_state *)opaque;



    switch (offset) {

    case 0x00: /* ID */

        return s->sys_id;

    case 0x04: /* SW */

        /* General purpose hardware switches.

           We don't have a useful way of exposing these to the user.  */

        return 0;

    case 0x08: /* LED */

        return s->leds;

    case 0x20: /* LOCK */

        return s->lockval;

    case 0x0c: /* OSC0 */

    case 0x10: /* OSC1 */

    case 0x14: /* OSC2 */

    case 0x18: /* OSC3 */

    case 0x1c: /* OSC4 */

    case 0x24: /* 100HZ */

        /* ??? Implement these.  */

        return 0;

    case 0x28: /* CFGDATA1 */

        return s->cfgdata1;

    case 0x2c: /* CFGDATA2 */

        return s->cfgdata2;

    case 0x30: /* FLAGS */

        return s->flags;

    case 0x38: /* NVFLAGS */

        return s->nvflags;

    case 0x40: /* RESETCTL */

        if (board_id(s) == BOARD_ID_VEXPRESS) {

            /* reserved: RAZ/WI */

            return 0;

        }

        return s->resetlevel;

    case 0x44: /* PCICTL */

        return 1;

    case 0x48: /* MCI */

        return s->sys_mci;

    case 0x4c: /* FLASH */

        return 0;

    case 0x50: /* CLCD */

        return s->sys_clcd;

    case 0x54: /* CLCDSER */

        return 0;

    case 0x58: /* BOOTCS */

        return 0;

    case 0x5c: /* 24MHz */

        return muldiv64(qemu_get_clock_ns(vm_clock), 24000000, get_ticks_per_sec());

    case 0x60: /* MISC */

        return 0;

    case 0x84: /* PROCID0 */

        return s->proc_id;

    case 0x88: /* PROCID1 */

        return 0xff000000;

    case 0x64: /* DMAPSR0 */

    case 0x68: /* DMAPSR1 */

    case 0x6c: /* DMAPSR2 */

    case 0x70: /* IOSEL */

    case 0x74: /* PLDCTL */

    case 0x80: /* BUSID */

    case 0x8c: /* OSCRESET0 */

    case 0x90: /* OSCRESET1 */

    case 0x94: /* OSCRESET2 */

    case 0x98: /* OSCRESET3 */

    case 0x9c: /* OSCRESET4 */

    case 0xc0: /* SYS_TEST_OSC0 */

    case 0xc4: /* SYS_TEST_OSC1 */

    case 0xc8: /* SYS_TEST_OSC2 */

    case 0xcc: /* SYS_TEST_OSC3 */

    case 0xd0: /* SYS_TEST_OSC4 */

        return 0;

    case 0xa0: /* SYS_CFGDATA */

        if (board_id(s) != BOARD_ID_VEXPRESS) {

            goto bad_reg;

        }

        return s->sys_cfgdata;

    case 0xa4: /* SYS_CFGCTRL */

        if (board_id(s) != BOARD_ID_VEXPRESS) {

            goto bad_reg;

        }

        return s->sys_cfgctrl;

    case 0xa8: /* SYS_CFGSTAT */

        if (board_id(s) != BOARD_ID_VEXPRESS) {

            goto bad_reg;

        }

        return s->sys_cfgstat;

    default:

    bad_reg:

        printf ("arm_sysctl_read: Bad register offset 0x%x\n", (int)offset);

        return 0;

    }

}
