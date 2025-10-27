static void omap_disc_write(void *opaque, hwaddr addr,

                            uint64_t value, unsigned size)

{

    struct omap_dss_s *s = (struct omap_dss_s *) opaque;



    if (size != 4) {

        omap_badwidth_write32(opaque, addr, value);

        return;

    }



    switch (addr) {

    case 0x010:	/* DISPC_SYSCONFIG */

        if (value & 2)						/* SOFTRESET */

            omap_dss_reset(s);

        s->dispc.idlemode = value & 0x301b;

        break;



    case 0x018:	/* DISPC_IRQSTATUS */

        s->dispc.irqst &= ~value;

        omap_dispc_interrupt_update(s);

        break;



    case 0x01c:	/* DISPC_IRQENABLE */

        s->dispc.irqen = value & 0xffff;

        omap_dispc_interrupt_update(s);

        break;



    case 0x040:	/* DISPC_CONTROL */

        s->dispc.control = value & 0x07ff9fff;

        s->dig.enable = (value >> 1) & 1;

        s->lcd.enable = (value >> 0) & 1;

        if (value & (1 << 12))			/* OVERLAY_OPTIMIZATION */

            if (!((s->dispc.l[1].attr | s->dispc.l[2].attr) & 1)) {

                fprintf(stderr, "%s: Overlay Optimization when no overlay "

                        "region effectively exists leads to "

                        "unpredictable behaviour!\n", __func__);

            }

        if (value & (1 << 6)) {				/* GODIGITAL */

            /* XXX: Shadowed fields are:

             * s->dispc.config

             * s->dispc.capable

             * s->dispc.bg[0]

             * s->dispc.bg[1]

             * s->dispc.trans[0]

             * s->dispc.trans[1]

             * s->dispc.line

             * s->dispc.timing[0]

             * s->dispc.timing[1]

             * s->dispc.timing[2]

             * s->dispc.timing[3]

             * s->lcd.nx

             * s->lcd.ny

             * s->dig.nx

             * s->dig.ny

             * s->dispc.l[0].addr[0]

             * s->dispc.l[0].addr[1]

             * s->dispc.l[0].addr[2]

             * s->dispc.l[0].posx

             * s->dispc.l[0].posy

             * s->dispc.l[0].nx

             * s->dispc.l[0].ny

             * s->dispc.l[0].tresh

             * s->dispc.l[0].rowinc

             * s->dispc.l[0].colinc

             * s->dispc.l[0].wininc

             * All they need to be loaded here from their shadow registers.

             */

        }

        if (value & (1 << 5)) {				/* GOLCD */

             /* XXX: Likewise for LCD here.  */

        }

        s->dispc.invalidate = 1;

        break;



    case 0x044:	/* DISPC_CONFIG */

        s->dispc.config = value & 0x3fff;

        /* XXX:

         * bits 2:1 (LOADMODE) reset to 0 after set to 1 and palette loaded

         * bits 2:1 (LOADMODE) reset to 2 after set to 3 and palette loaded

         */

        s->dispc.invalidate = 1;

        break;



    case 0x048:	/* DISPC_CAPABLE */

        s->dispc.capable = value & 0x3ff;

        break;



    case 0x04c:	/* DISPC_DEFAULT_COLOR0 */

        s->dispc.bg[0] = value & 0xffffff;

        s->dispc.invalidate = 1;

        break;

    case 0x050:	/* DISPC_DEFAULT_COLOR1 */

        s->dispc.bg[1] = value & 0xffffff;

        s->dispc.invalidate = 1;

        break;

    case 0x054:	/* DISPC_TRANS_COLOR0 */

        s->dispc.trans[0] = value & 0xffffff;

        s->dispc.invalidate = 1;

        break;

    case 0x058:	/* DISPC_TRANS_COLOR1 */

        s->dispc.trans[1] = value & 0xffffff;

        s->dispc.invalidate = 1;

        break;



    case 0x060:	/* DISPC_LINE_NUMBER */

        s->dispc.line = value & 0x7ff;

        break;



    case 0x064:	/* DISPC_TIMING_H */

        s->dispc.timing[0] = value & 0x0ff0ff3f;

        break;

    case 0x068:	/* DISPC_TIMING_V */

        s->dispc.timing[1] = value & 0x0ff0ff3f;

        break;

    case 0x06c:	/* DISPC_POL_FREQ */

        s->dispc.timing[2] = value & 0x0003ffff;

        break;

    case 0x070:	/* DISPC_DIVISOR */

        s->dispc.timing[3] = value & 0x00ff00ff;

        break;



    case 0x078:	/* DISPC_SIZE_DIG */

        s->dig.nx = ((value >>  0) & 0x7ff) + 1;		/* PPL */

        s->dig.ny = ((value >> 16) & 0x7ff) + 1;		/* LPP */

        s->dispc.invalidate = 1;

        break;

    case 0x07c:	/* DISPC_SIZE_LCD */

        s->lcd.nx = ((value >>  0) & 0x7ff) + 1;		/* PPL */

        s->lcd.ny = ((value >> 16) & 0x7ff) + 1;		/* LPP */

        s->dispc.invalidate = 1;

        break;

    case 0x080:	/* DISPC_GFX_BA0 */

        s->dispc.l[0].addr[0] = (hwaddr) value;

        s->dispc.invalidate = 1;

        break;

    case 0x084:	/* DISPC_GFX_BA1 */

        s->dispc.l[0].addr[1] = (hwaddr) value;

        s->dispc.invalidate = 1;

        break;

    case 0x088:	/* DISPC_GFX_POSITION */

        s->dispc.l[0].posx = ((value >>  0) & 0x7ff);		/* GFXPOSX */

        s->dispc.l[0].posy = ((value >> 16) & 0x7ff);		/* GFXPOSY */

        s->dispc.invalidate = 1;

        break;

    case 0x08c:	/* DISPC_GFX_SIZE */

        s->dispc.l[0].nx = ((value >>  0) & 0x7ff) + 1;		/* GFXSIZEX */

        s->dispc.l[0].ny = ((value >> 16) & 0x7ff) + 1;		/* GFXSIZEY */

        s->dispc.invalidate = 1;

        break;

    case 0x0a0:	/* DISPC_GFX_ATTRIBUTES */

        s->dispc.l[0].attr = value & 0x7ff;

        if (value & (3 << 9))

            fprintf(stderr, "%s: Big-endian pixel format not supported\n",

                            __FUNCTION__);

        s->dispc.l[0].enable = value & 1;

        s->dispc.l[0].bpp = (value >> 1) & 0xf;

        s->dispc.invalidate = 1;

        break;

    case 0x0a4:	/* DISPC_GFX_FIFO_TRESHOLD */

        s->dispc.l[0].tresh = value & 0x01ff01ff;

        break;

    case 0x0ac:	/* DISPC_GFX_ROW_INC */

        s->dispc.l[0].rowinc = value;

        s->dispc.invalidate = 1;

        break;

    case 0x0b0:	/* DISPC_GFX_PIXEL_INC */

        s->dispc.l[0].colinc = value;

        s->dispc.invalidate = 1;

        break;

    case 0x0b4:	/* DISPC_GFX_WINDOW_SKIP */

        s->dispc.l[0].wininc = value;

        break;

    case 0x0b8:	/* DISPC_GFX_TABLE_BA */

        s->dispc.l[0].addr[2] = (hwaddr) value;

        s->dispc.invalidate = 1;

        break;



    case 0x0bc:	/* DISPC_VID1_BA0 */

    case 0x0c0:	/* DISPC_VID1_BA1 */

    case 0x0c4:	/* DISPC_VID1_POSITION */

    case 0x0c8:	/* DISPC_VID1_SIZE */

    case 0x0cc:	/* DISPC_VID1_ATTRIBUTES */

    case 0x0d0:	/* DISPC_VID1_FIFO_TRESHOLD */

    case 0x0d8:	/* DISPC_VID1_ROW_INC */

    case 0x0dc:	/* DISPC_VID1_PIXEL_INC */

    case 0x0e0:	/* DISPC_VID1_FIR */

    case 0x0e4:	/* DISPC_VID1_PICTURE_SIZE */

    case 0x0e8:	/* DISPC_VID1_ACCU0 */

    case 0x0ec:	/* DISPC_VID1_ACCU1 */

    case 0x0f0 ... 0x140:	/* DISPC_VID1_FIR_COEF, DISPC_VID1_CONV_COEF */

    case 0x14c:	/* DISPC_VID2_BA0 */

    case 0x150:	/* DISPC_VID2_BA1 */

    case 0x154:	/* DISPC_VID2_POSITION */

    case 0x158:	/* DISPC_VID2_SIZE */

    case 0x15c:	/* DISPC_VID2_ATTRIBUTES */

    case 0x160:	/* DISPC_VID2_FIFO_TRESHOLD */

    case 0x168:	/* DISPC_VID2_ROW_INC */

    case 0x16c:	/* DISPC_VID2_PIXEL_INC */

    case 0x170:	/* DISPC_VID2_FIR */

    case 0x174:	/* DISPC_VID2_PICTURE_SIZE */

    case 0x178:	/* DISPC_VID2_ACCU0 */

    case 0x17c:	/* DISPC_VID2_ACCU1 */

    case 0x180 ... 0x1d0:	/* DISPC_VID2_FIR_COEF, DISPC_VID2_CONV_COEF */

    case 0x1d4:	/* DISPC_DATA_CYCLE1 */

    case 0x1d8:	/* DISPC_DATA_CYCLE2 */

    case 0x1dc:	/* DISPC_DATA_CYCLE3 */

        break;



    default:

        OMAP_BAD_REG(addr);

    }

}
