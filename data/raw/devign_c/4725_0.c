static uint32_t mipid_txrx(void *opaque, uint32_t cmd, int len)

{

    struct mipid_s *s = (struct mipid_s *) opaque;

    uint8_t ret;



    if (len > 9) {

        hw_error("%s: FIXME: bad SPI word width %i\n", __FUNCTION__, len);

    }



    if (s->p >= ARRAY_SIZE(s->resp)) {

        ret = 0;

    } else {

        ret = s->resp[s->p++];

    }

    if (s->pm-- > 0) {

        s->param[s->pm] = cmd;

    } else {

        s->cmd = cmd;

    }



    switch (s->cmd) {

    case 0x00:	/* NOP */

        break;



    case 0x01:	/* SWRESET */

        mipid_reset(s);

        break;



    case 0x02:	/* BSTROFF */

        s->booster = 0;

        break;

    case 0x03:	/* BSTRON */

        s->booster = 1;

        break;



    case 0x04:	/* RDDID */

        s->p = 0;

        s->resp[0] = (s->id >> 16) & 0xff;

        s->resp[1] = (s->id >>  8) & 0xff;

        s->resp[2] = (s->id >>  0) & 0xff;

        break;



    case 0x06:	/* RD_RED */

    case 0x07:	/* RD_GREEN */

        /* XXX the bootloader sometimes issues RD_BLUE meaning RDDID so

         * for the bootloader one needs to change this.  */

    case 0x08:	/* RD_BLUE */

        s->p = 0;

        /* TODO: return first pixel components */

        s->resp[0] = 0x01;

        break;



    case 0x09:	/* RDDST */

        s->p = 0;

        s->resp[0] = s->booster << 7;

        s->resp[1] = (5 << 4) | (s->partial << 2) |

                (s->sleep << 1) | s->normal;

        s->resp[2] = (s->vscr << 7) | (s->invert << 5) |

                (s->onoff << 2) | (s->te << 1) | (s->gamma >> 2);

        s->resp[3] = s->gamma << 6;

        break;



    case 0x0a:	/* RDDPM */

        s->p = 0;

        s->resp[0] = (s->onoff << 2) | (s->normal << 3) | (s->sleep << 4) |

                (s->partial << 5) | (s->sleep << 6) | (s->booster << 7);

        break;

    case 0x0b:	/* RDDMADCTR */

        s->p = 0;

        s->resp[0] = 0;

        break;

    case 0x0c:	/* RDDCOLMOD */

        s->p = 0;

        s->resp[0] = 5;	/* 65K colours */

        break;

    case 0x0d:	/* RDDIM */

        s->p = 0;

        s->resp[0] = (s->invert << 5) | (s->vscr << 7) | s->gamma;

        break;

    case 0x0e:	/* RDDSM */

        s->p = 0;

        s->resp[0] = s->te << 7;

        break;

    case 0x0f:	/* RDDSDR */

        s->p = 0;

        s->resp[0] = s->selfcheck;

        break;



    case 0x10:	/* SLPIN */

        s->sleep = 1;

        break;

    case 0x11:	/* SLPOUT */

        s->sleep = 0;

        s->selfcheck ^= 1 << 6;	/* POFF self-diagnosis Ok */

        break;



    case 0x12:	/* PTLON */

        s->partial = 1;

        s->normal = 0;

        s->vscr = 0;

        break;

    case 0x13:	/* NORON */

        s->partial = 0;

        s->normal = 1;

        s->vscr = 0;

        break;



    case 0x20:	/* INVOFF */

        s->invert = 0;

        break;

    case 0x21:	/* INVON */

        s->invert = 1;

        break;



    case 0x22:	/* APOFF */

    case 0x23:	/* APON */

        goto bad_cmd;



    case 0x25:	/* WRCNTR */

        if (s->pm < 0) {

            s->pm = 1;

        }

        goto bad_cmd;



    case 0x26:	/* GAMSET */

        if (!s->pm) {

            s->gamma = ctz32(s->param[0] & 0xf);

            if (s->gamma == 32) {

                s->gamma = -1; /* XXX: should this be 0? */

            }

        } else if (s->pm < 0) {

            s->pm = 1;

        }

        break;



    case 0x28:	/* DISPOFF */

        s->onoff = 0;

        break;

    case 0x29:	/* DISPON */

        s->onoff = 1;

        break;



    case 0x2a:	/* CASET */

    case 0x2b:	/* RASET */

    case 0x2c:	/* RAMWR */

    case 0x2d:	/* RGBSET */

    case 0x2e:	/* RAMRD */

    case 0x30:	/* PTLAR */

    case 0x33:	/* SCRLAR */

        goto bad_cmd;



    case 0x34:	/* TEOFF */

        s->te = 0;

        break;

    case 0x35:	/* TEON */

        if (!s->pm) {

            s->te = 1;

        } else if (s->pm < 0) {

            s->pm = 1;

        }

        break;



    case 0x36:	/* MADCTR */

        goto bad_cmd;



    case 0x37:	/* VSCSAD */

        s->partial = 0;

        s->normal = 0;

        s->vscr = 1;

        break;



    case 0x38:	/* IDMOFF */

    case 0x39:	/* IDMON */

    case 0x3a:	/* COLMOD */

        goto bad_cmd;



    case 0xb0:	/* CLKINT / DISCTL */

    case 0xb1:	/* CLKEXT */

        if (s->pm < 0) {

            s->pm = 2;

        }

        break;



    case 0xb4:	/* FRMSEL */

        break;



    case 0xb5:	/* FRM8SEL */

    case 0xb6:	/* TMPRNG / INIESC */

    case 0xb7:	/* TMPHIS / NOP2 */

    case 0xb8:	/* TMPREAD / MADCTL */

    case 0xba:	/* DISTCTR */

    case 0xbb:	/* EPVOL */

        goto bad_cmd;



    case 0xbd:	/* Unknown */

        s->p = 0;

        s->resp[0] = 0;

        s->resp[1] = 1;

        break;



    case 0xc2:	/* IFMOD */

        if (s->pm < 0) {

            s->pm = 2;

        }

        break;



    case 0xc6:	/* PWRCTL */

    case 0xc7:	/* PPWRCTL */

    case 0xd0:	/* EPWROUT */

    case 0xd1:	/* EPWRIN */

    case 0xd4:	/* RDEV */

    case 0xd5:	/* RDRR */

        goto bad_cmd;



    case 0xda:	/* RDID1 */

        s->p = 0;

        s->resp[0] = (s->id >> 16) & 0xff;

        break;

    case 0xdb:	/* RDID2 */

        s->p = 0;

        s->resp[0] = (s->id >>  8) & 0xff;

        break;

    case 0xdc:	/* RDID3 */

        s->p = 0;

        s->resp[0] = (s->id >>  0) & 0xff;

        break;



    default:

    bad_cmd:

        qemu_log_mask(LOG_GUEST_ERROR,

                      "%s: unknown command %02x\n", __func__, s->cmd);

        break;

    }



    return ret;

}
