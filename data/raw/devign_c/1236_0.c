static void musicpal_gpio_write(void *opaque, target_phys_addr_t offset,

                                uint64_t value, unsigned size)

{

    musicpal_gpio_state *s = opaque;

    switch (offset) {

    case MP_GPIO_OE_HI: /* used for LCD brightness control */

        s->lcd_brightness = (s->lcd_brightness & MP_GPIO_LCD_BRIGHTNESS) |

                         (value & MP_OE_LCD_BRIGHTNESS);

        musicpal_gpio_brightness_update(s);

        break;



    case MP_GPIO_OUT_LO:

        s->out_state = (s->out_state & 0xFFFF0000) | (value & 0xFFFF);

        break;

    case MP_GPIO_OUT_HI:

        s->out_state = (s->out_state & 0xFFFF) | (value << 16);

        s->lcd_brightness = (s->lcd_brightness & 0xFFFF) |

                            (s->out_state & MP_GPIO_LCD_BRIGHTNESS);

        musicpal_gpio_brightness_update(s);

        qemu_set_irq(s->out[3], (s->out_state >> MP_GPIO_I2C_DATA_BIT) & 1);

        qemu_set_irq(s->out[4], (s->out_state >> MP_GPIO_I2C_CLOCK_BIT) & 1);

        break;



    case MP_GPIO_IER_LO:

        s->ier = (s->ier & 0xFFFF0000) | (value & 0xFFFF);

        break;

    case MP_GPIO_IER_HI:

        s->ier = (s->ier & 0xFFFF) | (value << 16);

        break;



    case MP_GPIO_IMR_LO:

        s->imr = (s->imr & 0xFFFF0000) | (value & 0xFFFF);

        break;

    case MP_GPIO_IMR_HI:

        s->imr = (s->imr & 0xFFFF) | (value << 16);

        break;

    }

}
