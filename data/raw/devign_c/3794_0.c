static uint16_t blizzard_reg_read(void *opaque, uint8_t reg)

{

    BlizzardState *s = (BlizzardState *) opaque;



    switch (reg) {

    case 0x00:	/* Revision Code */

        return 0xa5;



    case 0x02:	/* Configuration Readback */

        return 0x83;	/* Macrovision OK, CNF[2:0] = 3 */



    case 0x04:	/* PLL M-Divider */

        return (s->pll - 1) | (1 << 7);

    case 0x06:	/* PLL Lock Range Control */

        return s->pll_range;

    case 0x08:	/* PLL Lock Synthesis Control 0 */

        return s->pll_ctrl & 0xff;

    case 0x0a:	/* PLL Lock Synthesis Control 1 */

        return s->pll_ctrl >> 8;

    case 0x0c:	/* PLL Mode Control 0 */

        return s->pll_mode;



    case 0x0e:	/* Clock-Source Select */

        return s->clksel;



    case 0x10:	/* Memory Controller Activate */

    case 0x14:	/* Memory Controller Bank 0 Status Flag */

        return s->memenable;



    case 0x18:	/* Auto-Refresh Interval Setting 0 */

        return s->memrefresh & 0xff;

    case 0x1a:	/* Auto-Refresh Interval Setting 1 */

        return s->memrefresh >> 8;



    case 0x1c:	/* Power-On Sequence Timing Control */

        return s->timing[0];

    case 0x1e:	/* Timing Control 0 */

        return s->timing[1];

    case 0x20:	/* Timing Control 1 */

        return s->timing[2];



    case 0x24:	/* Arbitration Priority Control */

        return s->priority;



    case 0x28:	/* LCD Panel Configuration */

        return s->lcd_config;



    case 0x2a:	/* LCD Horizontal Display Width */

        return s->x >> 3;

    case 0x2c:	/* LCD Horizontal Non-display Period */

        return s->hndp;

    case 0x2e:	/* LCD Vertical Display Height 0 */

        return s->y & 0xff;

    case 0x30:	/* LCD Vertical Display Height 1 */

        return s->y >> 8;

    case 0x32:	/* LCD Vertical Non-display Period */

        return s->vndp;

    case 0x34:	/* LCD HS Pulse-width */

        return s->hsync;

    case 0x36:	/* LCd HS Pulse Start Position */

        return s->skipx >> 3;

    case 0x38:	/* LCD VS Pulse-width */

        return s->vsync;

    case 0x3a:	/* LCD VS Pulse Start Position */

        return s->skipy;



    case 0x3c:	/* PCLK Polarity */

        return s->pclk;



    case 0x3e:	/* High-speed Serial Interface Tx Configuration Port 0 */

        return s->hssi_config[0];

    case 0x40:	/* High-speed Serial Interface Tx Configuration Port 1 */

        return s->hssi_config[1];

    case 0x42:	/* High-speed Serial Interface Tx Mode */

        return s->hssi_config[2];

    case 0x44:	/* TV Display Configuration */

        return s->tv_config;

    case 0x46 ... 0x4c:	/* TV Vertical Blanking Interval Data bits */

        return s->tv_timing[(reg - 0x46) >> 1];

    case 0x4e:	/* VBI: Closed Caption / XDS Control / Status */

        return s->vbi;

    case 0x50:	/* TV Horizontal Start Position */

        return s->tv_x;

    case 0x52:	/* TV Vertical Start Position */

        return s->tv_y;

    case 0x54:	/* TV Test Pattern Setting */

        return s->tv_test;

    case 0x56:	/* TV Filter Setting */

        return s->tv_filter_config;

    case 0x58:	/* TV Filter Coefficient Index */

        return s->tv_filter_idx;

    case 0x5a:	/* TV Filter Coefficient Data */

        if (s->tv_filter_idx < 0x20)

            return s->tv_filter_coeff[s->tv_filter_idx ++];

        return 0;



    case 0x60:	/* Input YUV/RGB Translate Mode 0 */

        return s->yrc[0];

    case 0x62:	/* Input YUV/RGB Translate Mode 1 */

        return s->yrc[1];

    case 0x64:	/* U Data Fix */

        return s->u;

    case 0x66:	/* V Data Fix */

        return s->v;



    case 0x68:	/* Display Mode */

        return s->mode;



    case 0x6a:	/* Special Effects */

        return s->effect;



    case 0x6c:	/* Input Window X Start Position 0 */

        return s->ix[0] & 0xff;

    case 0x6e:	/* Input Window X Start Position 1 */

        return s->ix[0] >> 3;

    case 0x70:	/* Input Window Y Start Position 0 */

        return s->ix[0] & 0xff;

    case 0x72:	/* Input Window Y Start Position 1 */

        return s->ix[0] >> 3;

    case 0x74:	/* Input Window X End Position 0 */

        return s->ix[1] & 0xff;

    case 0x76:	/* Input Window X End Position 1 */

        return s->ix[1] >> 3;

    case 0x78:	/* Input Window Y End Position 0 */

        return s->ix[1] & 0xff;

    case 0x7a:	/* Input Window Y End Position 1 */

        return s->ix[1] >> 3;

    case 0x7c:	/* Output Window X Start Position 0 */

        return s->ox[0] & 0xff;

    case 0x7e:	/* Output Window X Start Position 1 */

        return s->ox[0] >> 3;

    case 0x80:	/* Output Window Y Start Position 0 */

        return s->oy[0] & 0xff;

    case 0x82:	/* Output Window Y Start Position 1 */

        return s->oy[0] >> 3;

    case 0x84:	/* Output Window X End Position 0 */

        return s->ox[1] & 0xff;

    case 0x86:	/* Output Window X End Position 1 */

        return s->ox[1] >> 3;

    case 0x88:	/* Output Window Y End Position 0 */

        return s->oy[1] & 0xff;

    case 0x8a:	/* Output Window Y End Position 1 */

        return s->oy[1] >> 3;



    case 0x8c:	/* Input Data Format */

        return s->iformat;

    case 0x8e:	/* Data Source Select */

        return s->source;

    case 0x90:	/* Display Memory Data Port */

        return 0;



    case 0xa8:	/* Border Color 0 */

        return s->border_r;

    case 0xaa:	/* Border Color 1 */

        return s->border_g;

    case 0xac:	/* Border Color 2 */

        return s->border_b;



    case 0xb4:	/* Gamma Correction Enable */

        return s->gamma_config;

    case 0xb6:	/* Gamma Correction Table Index */

        return s->gamma_idx;

    case 0xb8:	/* Gamma Correction Table Data */

        return s->gamma_lut[s->gamma_idx ++];



    case 0xba:	/* 3x3 Matrix Enable */

        return s->matrix_ena;

    case 0xbc ... 0xde:	/* Coefficient Registers */

        return s->matrix_coeff[(reg - 0xbc) >> 1];

    case 0xe0:	/* 3x3 Matrix Red Offset */

        return s->matrix_r;

    case 0xe2:	/* 3x3 Matrix Green Offset */

        return s->matrix_g;

    case 0xe4:	/* 3x3 Matrix Blue Offset */

        return s->matrix_b;



    case 0xe6:	/* Power-save */

        return s->pm;

    case 0xe8:	/* Non-display Period Control / Status */

        return s->status | (1 << 5);

    case 0xea:	/* RGB Interface Control */

        return s->rgbgpio_dir;

    case 0xec:	/* RGB Interface Status */

        return s->rgbgpio;

    case 0xee:	/* General-purpose IO Pins Configuration */

        return s->gpio_dir;

    case 0xf0:	/* General-purpose IO Pins Status / Control */

        return s->gpio;

    case 0xf2:	/* GPIO Positive Edge Interrupt Trigger */

        return s->gpio_edge[0];

    case 0xf4:	/* GPIO Negative Edge Interrupt Trigger */

        return s->gpio_edge[1];

    case 0xf6:	/* GPIO Interrupt Status */

        return s->gpio_irq;

    case 0xf8:	/* GPIO Pull-down Control */

        return s->gpio_pdown;



    default:

        fprintf(stderr, "%s: unknown register %02x\n", __FUNCTION__, reg);

        return 0;

    }

}
