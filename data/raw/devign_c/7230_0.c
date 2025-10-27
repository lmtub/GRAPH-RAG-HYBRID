static void blizzard_reg_write(void *opaque, uint8_t reg, uint16_t value)

{

    BlizzardState *s = (BlizzardState *) opaque;



    switch (reg) {

    case 0x04:	/* PLL M-Divider */

        s->pll = (value & 0x3f) + 1;

        break;

    case 0x06:	/* PLL Lock Range Control */

        s->pll_range = value & 3;

        break;

    case 0x08:	/* PLL Lock Synthesis Control 0 */

        s->pll_ctrl &= 0xf00;

        s->pll_ctrl |= (value << 0) & 0x0ff;

        break;

    case 0x0a:	/* PLL Lock Synthesis Control 1 */

        s->pll_ctrl &= 0x0ff;

        s->pll_ctrl |= (value << 8) & 0xf00;

        break;

    case 0x0c:	/* PLL Mode Control 0 */

        s->pll_mode = value & 0x77;

        if ((value & 3) == 0 || (value & 3) == 3)

            fprintf(stderr, "%s: wrong PLL Control bits (%i)\n",

                    __FUNCTION__, value & 3);

        break;



    case 0x0e:	/* Clock-Source Select */

        s->clksel = value & 0xff;

        break;



    case 0x10:	/* Memory Controller Activate */

        s->memenable = value & 1;

        break;

    case 0x14:	/* Memory Controller Bank 0 Status Flag */

        break;



    case 0x18:	/* Auto-Refresh Interval Setting 0 */

        s->memrefresh &= 0xf00;

        s->memrefresh |= (value << 0) & 0x0ff;

        break;

    case 0x1a:	/* Auto-Refresh Interval Setting 1 */

        s->memrefresh &= 0x0ff;

        s->memrefresh |= (value << 8) & 0xf00;

        break;



    case 0x1c:	/* Power-On Sequence Timing Control */

        s->timing[0] = value & 0x7f;

        break;

    case 0x1e:	/* Timing Control 0 */

        s->timing[1] = value & 0x17;

        break;

    case 0x20:	/* Timing Control 1 */

        s->timing[2] = value & 0x35;

        break;



    case 0x24:	/* Arbitration Priority Control */

        s->priority = value & 1;

        break;



    case 0x28:	/* LCD Panel Configuration */

        s->lcd_config = value & 0xff;

        if (value & (1 << 7))

            fprintf(stderr, "%s: data swap not supported!\n", __FUNCTION__);

        break;



    case 0x2a:	/* LCD Horizontal Display Width */

        s->x = value << 3;

        break;

    case 0x2c:	/* LCD Horizontal Non-display Period */

        s->hndp = value & 0xff;

        break;

    case 0x2e:	/* LCD Vertical Display Height 0 */

        s->y &= 0x300;

        s->y |= (value << 0) & 0x0ff;

        break;

    case 0x30:	/* LCD Vertical Display Height 1 */

        s->y &= 0x0ff;

        s->y |= (value << 8) & 0x300;

        break;

    case 0x32:	/* LCD Vertical Non-display Period */

        s->vndp = value & 0xff;

        break;

    case 0x34:	/* LCD HS Pulse-width */

        s->hsync = value & 0xff;

        break;

    case 0x36:	/* LCD HS Pulse Start Position */

        s->skipx = value & 0xff;

        break;

    case 0x38:	/* LCD VS Pulse-width */

        s->vsync = value & 0xbf;

        break;

    case 0x3a:	/* LCD VS Pulse Start Position */

        s->skipy = value & 0xff;

        break;



    case 0x3c:	/* PCLK Polarity */

        s->pclk = value & 0x82;

        /* Affects calculation of s->hndp, s->hsync and s->skipx.  */

        break;



    case 0x3e:	/* High-speed Serial Interface Tx Configuration Port 0 */

        s->hssi_config[0] = value;

        break;

    case 0x40:	/* High-speed Serial Interface Tx Configuration Port 1 */

        s->hssi_config[1] = value;

        if (((value >> 4) & 3) == 3)

            fprintf(stderr, "%s: Illegal active-data-links value\n",

                            __FUNCTION__);

        break;

    case 0x42:	/* High-speed Serial Interface Tx Mode */

        s->hssi_config[2] = value & 0xbd;

        break;



    case 0x44:	/* TV Display Configuration */

        s->tv_config = value & 0xfe;

        break;

    case 0x46 ... 0x4c:	/* TV Vertical Blanking Interval Data bits 0 */

        s->tv_timing[(reg - 0x46) >> 1] = value;

        break;

    case 0x4e:	/* VBI: Closed Caption / XDS Control / Status */

        s->vbi = value;

        break;

    case 0x50:	/* TV Horizontal Start Position */

        s->tv_x = value;

        break;

    case 0x52:	/* TV Vertical Start Position */

        s->tv_y = value & 0x7f;

        break;

    case 0x54:	/* TV Test Pattern Setting */

        s->tv_test = value;

        break;

    case 0x56:	/* TV Filter Setting */

        s->tv_filter_config = value & 0xbf;

        break;

    case 0x58:	/* TV Filter Coefficient Index */

        s->tv_filter_idx = value & 0x1f;

        break;

    case 0x5a:	/* TV Filter Coefficient Data */

        if (s->tv_filter_idx < 0x20)

            s->tv_filter_coeff[s->tv_filter_idx ++] = value;

        break;



    case 0x60:	/* Input YUV/RGB Translate Mode 0 */

        s->yrc[0] = value & 0xb0;

        break;

    case 0x62:	/* Input YUV/RGB Translate Mode 1 */

        s->yrc[1] = value & 0x30;

        break;

    case 0x64:	/* U Data Fix */

        s->u = value & 0xff;

        break;

    case 0x66:	/* V Data Fix */

        s->v = value & 0xff;

        break;



    case 0x68:	/* Display Mode */

        if ((s->mode ^ value) & 3)

            s->invalidate = 1;

        s->mode = value & 0xb7;

        s->enable = value & 1;

        s->blank = (value >> 1) & 1;

        if (value & (1 << 4))

            fprintf(stderr, "%s: Macrovision enable attempt!\n", __FUNCTION__);

        break;



    case 0x6a:	/* Special Effects */

        s->effect = value & 0xfb;

        break;



    case 0x6c:	/* Input Window X Start Position 0 */

        s->ix[0] &= 0x300;

        s->ix[0] |= (value << 0) & 0x0ff;

        break;

    case 0x6e:	/* Input Window X Start Position 1 */

        s->ix[0] &= 0x0ff;

        s->ix[0] |= (value << 8) & 0x300;

        break;

    case 0x70:	/* Input Window Y Start Position 0 */

        s->iy[0] &= 0x300;

        s->iy[0] |= (value << 0) & 0x0ff;

        break;

    case 0x72:	/* Input Window Y Start Position 1 */

        s->iy[0] &= 0x0ff;

        s->iy[0] |= (value << 8) & 0x300;

        break;

    case 0x74:	/* Input Window X End Position 0 */

        s->ix[1] &= 0x300;

        s->ix[1] |= (value << 0) & 0x0ff;

        break;

    case 0x76:	/* Input Window X End Position 1 */

        s->ix[1] &= 0x0ff;

        s->ix[1] |= (value << 8) & 0x300;

        break;

    case 0x78:	/* Input Window Y End Position 0 */

        s->iy[1] &= 0x300;

        s->iy[1] |= (value << 0) & 0x0ff;

        break;

    case 0x7a:	/* Input Window Y End Position 1 */

        s->iy[1] &= 0x0ff;

        s->iy[1] |= (value << 8) & 0x300;

        break;

    case 0x7c:	/* Output Window X Start Position 0 */

        s->ox[0] &= 0x300;

        s->ox[0] |= (value << 0) & 0x0ff;

        break;

    case 0x7e:	/* Output Window X Start Position 1 */

        s->ox[0] &= 0x0ff;

        s->ox[0] |= (value << 8) & 0x300;

        break;

    case 0x80:	/* Output Window Y Start Position 0 */

        s->oy[0] &= 0x300;

        s->oy[0] |= (value << 0) & 0x0ff;

        break;

    case 0x82:	/* Output Window Y Start Position 1 */

        s->oy[0] &= 0x0ff;

        s->oy[0] |= (value << 8) & 0x300;

        break;

    case 0x84:	/* Output Window X End Position 0 */

        s->ox[1] &= 0x300;

        s->ox[1] |= (value << 0) & 0x0ff;

        break;

    case 0x86:	/* Output Window X End Position 1 */

        s->ox[1] &= 0x0ff;

        s->ox[1] |= (value << 8) & 0x300;

        break;

    case 0x88:	/* Output Window Y End Position 0 */

        s->oy[1] &= 0x300;

        s->oy[1] |= (value << 0) & 0x0ff;

        break;

    case 0x8a:	/* Output Window Y End Position 1 */

        s->oy[1] &= 0x0ff;

        s->oy[1] |= (value << 8) & 0x300;

        break;



    case 0x8c:	/* Input Data Format */

        s->iformat = value & 0xf;

        s->bpp = blizzard_iformat_bpp[s->iformat];

        if (!s->bpp)

            fprintf(stderr, "%s: Illegal or unsupported input format %x\n",

                            __FUNCTION__, s->iformat);

        break;

    case 0x8e:	/* Data Source Select */

        s->source = value & 7;

        /* Currently all windows will be "destructive overlays".  */

        if ((!(s->effect & (1 << 3)) && (s->ix[0] != s->ox[0] ||

                                        s->iy[0] != s->oy[0] ||

                                        s->ix[1] != s->ox[1] ||

                                        s->iy[1] != s->oy[1])) ||

                        !((s->ix[1] - s->ix[0]) & (s->iy[1] - s->iy[0]) &

                          (s->ox[1] - s->ox[0]) & (s->oy[1] - s->oy[0]) & 1))

            fprintf(stderr, "%s: Illegal input/output window positions\n",

                            __FUNCTION__);



        blizzard_transfer_setup(s);

        break;



    case 0x90:	/* Display Memory Data Port */

        if (!s->data.len && !blizzard_transfer_setup(s))

            break;



        *s->data.ptr ++ = value;

        if (-- s->data.len == 0)

            blizzard_window(s);

        break;



    case 0xa8:	/* Border Color 0 */

        s->border_r = value;

        break;

    case 0xaa:	/* Border Color 1 */

        s->border_g = value;

        break;

    case 0xac:	/* Border Color 2 */

        s->border_b = value;

        break;



    case 0xb4:	/* Gamma Correction Enable */

        s->gamma_config = value & 0x87;

        break;

    case 0xb6:	/* Gamma Correction Table Index */

        s->gamma_idx = value;

        break;

    case 0xb8:	/* Gamma Correction Table Data */

        s->gamma_lut[s->gamma_idx ++] = value;

        break;



    case 0xba:	/* 3x3 Matrix Enable */

        s->matrix_ena = value & 1;

        break;

    case 0xbc ... 0xde:	/* Coefficient Registers */

        s->matrix_coeff[(reg - 0xbc) >> 1] = value & ((reg & 2) ? 0x80 : 0xff);

        break;

    case 0xe0:	/* 3x3 Matrix Red Offset */

        s->matrix_r = value;

        break;

    case 0xe2:	/* 3x3 Matrix Green Offset */

        s->matrix_g = value;

        break;

    case 0xe4:	/* 3x3 Matrix Blue Offset */

        s->matrix_b = value;

        break;



    case 0xe6:	/* Power-save */

        s->pm = value & 0x83;

        if (value & s->mode & 1)

            fprintf(stderr, "%s: The display must be disabled before entering "

                            "Standby Mode\n", __FUNCTION__);

        break;

    case 0xe8:	/* Non-display Period Control / Status */

        s->status = value & 0x1b;

        break;

    case 0xea:	/* RGB Interface Control */

        s->rgbgpio_dir = value & 0x8f;

        break;

    case 0xec:	/* RGB Interface Status */

        s->rgbgpio = value & 0xcf;

        break;

    case 0xee:	/* General-purpose IO Pins Configuration */

        s->gpio_dir = value;

        break;

    case 0xf0:	/* General-purpose IO Pins Status / Control */

        s->gpio = value;

        break;

    case 0xf2:	/* GPIO Positive Edge Interrupt Trigger */

        s->gpio_edge[0] = value;

        break;

    case 0xf4:	/* GPIO Negative Edge Interrupt Trigger */

        s->gpio_edge[1] = value;

        break;

    case 0xf6:	/* GPIO Interrupt Status */

        s->gpio_irq &= value;

        break;

    case 0xf8:	/* GPIO Pull-down Control */

        s->gpio_pdown = value;

        break;



    default:

        fprintf(stderr, "%s: unknown register %02x\n", __FUNCTION__, reg);

        break;

    }

}
