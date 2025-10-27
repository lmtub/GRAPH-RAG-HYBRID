static uint64_t pxa2xx_gpio_read(void *opaque, hwaddr offset,

                                 unsigned size)

{

    PXA2xxGPIOInfo *s = (PXA2xxGPIOInfo *) opaque;

    uint32_t ret;

    int bank;

    if (offset >= 0x200)

        return 0;



    bank = pxa2xx_gpio_regs[offset].bank;

    switch (pxa2xx_gpio_regs[offset].reg) {

    case GPDR:		/* GPIO Pin-Direction registers */

        return s->dir[bank];



    case GPSR:		/* GPIO Pin-Output Set registers */

        printf("%s: Read from a write-only register " REG_FMT "\n",

                        __FUNCTION__, offset);

        return s->gpsr[bank];	/* Return last written value.  */



    case GPCR:		/* GPIO Pin-Output Clear registers */

        printf("%s: Read from a write-only register " REG_FMT "\n",

                        __FUNCTION__, offset);

        return 31337;		/* Specified as unpredictable in the docs.  */



    case GRER:		/* GPIO Rising-Edge Detect Enable registers */

        return s->rising[bank];



    case GFER:		/* GPIO Falling-Edge Detect Enable registers */

        return s->falling[bank];



    case GAFR_L:	/* GPIO Alternate Function registers */

        return s->gafr[bank * 2];



    case GAFR_U:	/* GPIO Alternate Function registers */

        return s->gafr[bank * 2 + 1];



    case GPLR:		/* GPIO Pin-Level registers */

        ret = (s->olevel[bank] & s->dir[bank]) |

                (s->ilevel[bank] & ~s->dir[bank]);

        qemu_irq_raise(s->read_notify);

        return ret;



    case GEDR:		/* GPIO Edge Detect Status registers */

        return s->status[bank];



    default:

        hw_error("%s: Bad offset " REG_FMT "\n", __FUNCTION__, offset);

    }



    return 0;

}
