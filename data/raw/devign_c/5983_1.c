static void strongarm_gpio_write(void *opaque, hwaddr offset,

                                 uint64_t value, unsigned size)

{

    StrongARMGPIOInfo *s = opaque;



    switch (offset) {

    case GPDR:        /* GPIO Pin-Direction registers */

        s->dir = value;

        strongarm_gpio_handler_update(s);

        break;



    case GPSR:        /* GPIO Pin-Output Set registers */

        s->olevel |= value;

        strongarm_gpio_handler_update(s);

        s->gpsr = value;

        break;



    case GPCR:        /* GPIO Pin-Output Clear registers */

        s->olevel &= ~value;

        strongarm_gpio_handler_update(s);

        break;



    case GRER:        /* GPIO Rising-Edge Detect Enable registers */

        s->rising = value;

        break;



    case GFER:        /* GPIO Falling-Edge Detect Enable registers */

        s->falling = value;

        break;



    case GAFR:        /* GPIO Alternate Function registers */

        s->gafr = value;

        break;



    case GEDR:        /* GPIO Edge Detect Status registers */

        s->status &= ~value;

        strongarm_gpio_irq_update(s);

        break;



    default:

        printf("%s: Bad offset 0x" TARGET_FMT_plx "\n", __func__, offset);

    }

}
