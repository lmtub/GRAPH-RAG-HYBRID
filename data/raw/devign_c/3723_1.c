static void vbe_update_vgaregs(VGACommonState *s)

{

    int h, shift_control;



    if (!vbe_enabled(s)) {

        /* vbe is turned off -- nothing to do */

        return;

    }



    /* graphic mode + memory map 1 */

    s->gr[VGA_GFX_MISC] = (s->gr[VGA_GFX_MISC] & ~0x0c) | 0x04 |

        VGA_GR06_GRAPHICS_MODE;

    s->cr[VGA_CRTC_MODE] |= 3; /* no CGA modes */

    s->cr[VGA_CRTC_OFFSET] = s->vbe_line_offset >> 3;

    /* width */

    s->cr[VGA_CRTC_H_DISP] =

        (s->vbe_regs[VBE_DISPI_INDEX_XRES] >> 3) - 1;

    /* height (only meaningful if < 1024) */

    h = s->vbe_regs[VBE_DISPI_INDEX_YRES] - 1;

    s->cr[VGA_CRTC_V_DISP_END] = h;

    s->cr[VGA_CRTC_OVERFLOW] = (s->cr[VGA_CRTC_OVERFLOW] & ~0x42) |

        ((h >> 7) & 0x02) | ((h >> 3) & 0x40);

    /* line compare to 1023 */

    s->cr[VGA_CRTC_LINE_COMPARE] = 0xff;

    s->cr[VGA_CRTC_OVERFLOW] |= 0x10;

    s->cr[VGA_CRTC_MAX_SCAN] |= 0x40;



    if (s->vbe_regs[VBE_DISPI_INDEX_BPP] == 4) {

        shift_control = 0;

        s->sr[VGA_SEQ_CLOCK_MODE] &= ~8; /* no double line */

    } else {

        shift_control = 2;

        /* set chain 4 mode */

        s->sr[VGA_SEQ_MEMORY_MODE] |= VGA_SR04_CHN_4M;

        /* activate all planes */

        s->sr[VGA_SEQ_PLANE_WRITE] |= VGA_SR02_ALL_PLANES;

    }

    s->gr[VGA_GFX_MODE] = (s->gr[VGA_GFX_MODE] & ~0x60) |

        (shift_control << 5);

    s->cr[VGA_CRTC_MAX_SCAN] &= ~0x9f; /* no double scan */

}
