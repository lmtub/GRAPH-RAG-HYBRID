static void pxa2xx_gpio_write(void *opaque, hwaddr offset,

                              uint64_t value, unsigned size)

{

    PXA2xxGPIOInfo *s = (PXA2xxGPIOInfo *) opaque;

    int bank;

    if (offset >= 0x200)

        return;



    bank = pxa2xx_gpio_regs[offset].bank;

    switch (pxa2xx_gpio_regs[offset].reg) {

    case GPDR:		/* GPIO Pin-Direction registers */

        s->dir[bank] = value;

        pxa2xx_gpio_handler_update(s);

        break;



    case GPSR:		/* GPIO Pin-Output Set registers */

        s->olevel[bank] |= value;

        pxa2xx_gpio_handler_update(s);

        s->gpsr[bank] = value;

        break;



    case GPCR:		/* GPIO Pin-Output Clear registers */

        s->olevel[bank] &= ~value;

        pxa2xx_gpio_handler_update(s);

        break;



    case GRER:		/* GPIO Rising-Edge Detect Enable registers */

        s->rising[bank] = value;

        break;



    case GFER:		/* GPIO Falling-Edge Detect Enable registers */

        s->falling[bank] = value;

        break;



    case GAFR_L:	/* GPIO Alternate Function registers */

        s->gafr[bank * 2] = value;

        break;



    case GAFR_U:	/* GPIO Alternate Function registers */

        s->gafr[bank * 2 + 1] = value;

        break;



    case GEDR:		/* GPIO Edge Detect Status registers */

        s->status[bank] &= ~value;

        pxa2xx_gpio_irq_update(s);

        break;



    default:

        hw_error("%s: Bad offset " REG_FMT "\n", __FUNCTION__, offset);

    }

}
