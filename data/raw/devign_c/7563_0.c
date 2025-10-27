static void omap_sysctl_write(void *opaque, target_phys_addr_t addr,

                uint32_t value)

{

    struct omap_sysctl_s *s = (struct omap_sysctl_s *) opaque;



    switch (addr) {

    case 0x000:	/* CONTROL_REVISION */

    case 0x2a4:	/* CONTROL_MSUSPENDMUX_5 */

    case 0x2c0:	/* CONTROL_PSA_VALUE */

    case 0x2f8:	/* CONTROL_STATUS */

    case 0x2fc:	/* CONTROL_GENERAL_PURPOSE_STATUS */

    case 0x300:	/* CONTROL_RPUB_KEY_H_0 */

    case 0x304:	/* CONTROL_RPUB_KEY_H_1 */

    case 0x308:	/* CONTROL_RPUB_KEY_H_2 */

    case 0x30c:	/* CONTROL_RPUB_KEY_H_3 */

    case 0x310:	/* CONTROL_RAND_KEY_0 */

    case 0x314:	/* CONTROL_RAND_KEY_1 */

    case 0x318:	/* CONTROL_RAND_KEY_2 */

    case 0x31c:	/* CONTROL_RAND_KEY_3 */

    case 0x320:	/* CONTROL_CUST_KEY_0 */

    case 0x324:	/* CONTROL_CUST_KEY_1 */

    case 0x330:	/* CONTROL_TEST_KEY_0 */

    case 0x334:	/* CONTROL_TEST_KEY_1 */

    case 0x338:	/* CONTROL_TEST_KEY_2 */

    case 0x33c:	/* CONTROL_TEST_KEY_3 */

    case 0x340:	/* CONTROL_TEST_KEY_4 */

    case 0x344:	/* CONTROL_TEST_KEY_5 */

    case 0x348:	/* CONTROL_TEST_KEY_6 */

    case 0x34c:	/* CONTROL_TEST_KEY_7 */

    case 0x350:	/* CONTROL_TEST_KEY_8 */

    case 0x354:	/* CONTROL_TEST_KEY_9 */

        OMAP_RO_REG(addr);

        return;



    case 0x010:	/* CONTROL_SYSCONFIG */

        s->sysconfig = value & 0x1e;

        break;



    case 0x030 ... 0x140:	/* CONTROL_PADCONF - only used in the POP */

        /* XXX: should check constant bits */

        s->padconf[(addr - 0x30) >> 2] = value & 0x1f1f1f1f;

        break;



    case 0x270:	/* CONTROL_DEBOBS */

        s->obs = value & 0xff;

        break;



    case 0x274:	/* CONTROL_DEVCONF */

        s->devconfig = value & 0xffffc7ff;

        break;



    case 0x28c:	/* CONTROL_EMU_SUPPORT */

        break;



    case 0x290:	/* CONTROL_MSUSPENDMUX_0 */

        s->msuspendmux[0] = value & 0x3fffffff;

        break;

    case 0x294:	/* CONTROL_MSUSPENDMUX_1 */

        s->msuspendmux[1] = value & 0x3fffffff;

        break;

    case 0x298:	/* CONTROL_MSUSPENDMUX_2 */

        s->msuspendmux[2] = value & 0x3fffffff;

        break;

    case 0x29c:	/* CONTROL_MSUSPENDMUX_3 */

        s->msuspendmux[3] = value & 0x3fffffff;

        break;

    case 0x2a0:	/* CONTROL_MSUSPENDMUX_4 */

        s->msuspendmux[4] = value & 0x3fffffff;

        break;



    case 0x2b8:	/* CONTROL_PSA_CTRL */

        s->psaconfig = value & 0x1c;

        s->psaconfig |= (value & 0x20) ? 2 : 1;

        break;

    case 0x2bc:	/* CONTROL_PSA_CMD */

        break;



    case 0x2b0:	/* CONTROL_SEC_CTRL */

    case 0x2b4:	/* CONTROL_SEC_TEST */

    case 0x2d0:	/* CONTROL_SEC_EMU */

    case 0x2d4:	/* CONTROL_SEC_TAP */

    case 0x2d8:	/* CONTROL_OCM_RAM_PERM */

    case 0x2dc:	/* CONTROL_OCM_PUB_RAM_ADD */

    case 0x2e0:	/* CONTROL_EXT_SEC_RAM_START_ADD */

    case 0x2e4:	/* CONTROL_EXT_SEC_RAM_STOP_ADD */

    case 0x2f0:	/* CONTROL_SEC_STATUS */

    case 0x2f4:	/* CONTROL_SEC_ERR_STATUS */

        break;



    default:

        OMAP_BAD_REG(addr);

        return;

    }

}
