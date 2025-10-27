static void vga_update_text(void *opaque, console_ch_t *chardata)

{

    VGACommonState *s =  opaque;

    int graphic_mode, i, cursor_offset, cursor_visible;

    int cw, cheight, width, height, size, c_min, c_max;

    uint32_t *src;

    console_ch_t *dst, val;

    char msg_buffer[80];

    int full_update = 0;



    qemu_flush_coalesced_mmio_buffer();



    if (!(s->ar_index & 0x20)) {

        graphic_mode = GMODE_BLANK;

    } else {

        graphic_mode = s->gr[VGA_GFX_MISC] & VGA_GR06_GRAPHICS_MODE;

    }

    if (graphic_mode != s->graphic_mode) {

        s->graphic_mode = graphic_mode;

        full_update = 1;

    }

    if (s->last_width == -1) {

        s->last_width = 0;

        full_update = 1;

    }



    switch (graphic_mode) {

    case GMODE_TEXT:

        /* TODO: update palette */

        full_update |= update_basic_params(s);



        /* total width & height */

        cheight = (s->cr[VGA_CRTC_MAX_SCAN] & 0x1f) + 1;

        cw = 8;

        if (!(s->sr[VGA_SEQ_CLOCK_MODE] & VGA_SR01_CHAR_CLK_8DOTS)) {

            cw = 9;

        }

        if (s->sr[VGA_SEQ_CLOCK_MODE] & 0x08) {

            cw = 16; /* NOTE: no 18 pixel wide */

        }

        width = (s->cr[VGA_CRTC_H_DISP] + 1);

        if (s->cr[VGA_CRTC_V_TOTAL] == 100) {

            /* ugly hack for CGA 160x100x16 - explain me the logic */

            height = 100;

        } else {

            height = s->cr[VGA_CRTC_V_DISP_END] |

                ((s->cr[VGA_CRTC_OVERFLOW] & 0x02) << 7) |

                ((s->cr[VGA_CRTC_OVERFLOW] & 0x40) << 3);

            height = (height + 1) / cheight;

        }



        size = (height * width);

        if (size > CH_ATTR_SIZE) {

            if (!full_update)

                return;



            snprintf(msg_buffer, sizeof(msg_buffer), "%i x %i Text mode",

                     width, height);

            break;

        }



        if (width != s->last_width || height != s->last_height ||

            cw != s->last_cw || cheight != s->last_ch) {

            s->last_scr_width = width * cw;

            s->last_scr_height = height * cheight;

            qemu_console_resize(s->con, s->last_scr_width, s->last_scr_height);

            dpy_text_resize(s->con, width, height);

            s->last_depth = 0;

            s->last_width = width;

            s->last_height = height;

            s->last_ch = cheight;

            s->last_cw = cw;

            full_update = 1;

        }



        if (full_update) {

            s->full_update_gfx = 1;

        }

        if (s->full_update_text) {

            s->full_update_text = 0;

            full_update |= 1;

        }



        /* Update "hardware" cursor */

        cursor_offset = ((s->cr[VGA_CRTC_CURSOR_HI] << 8) |

                         s->cr[VGA_CRTC_CURSOR_LO]) - s->start_addr;

        if (cursor_offset != s->cursor_offset ||

            s->cr[VGA_CRTC_CURSOR_START] != s->cursor_start ||

            s->cr[VGA_CRTC_CURSOR_END] != s->cursor_end || full_update) {

            cursor_visible = !(s->cr[VGA_CRTC_CURSOR_START] & 0x20);

            if (cursor_visible && cursor_offset < size && cursor_offset >= 0)

                dpy_text_cursor(s->con,

                                TEXTMODE_X(cursor_offset),

                                TEXTMODE_Y(cursor_offset));

            else

                dpy_text_cursor(s->con, -1, -1);

            s->cursor_offset = cursor_offset;

            s->cursor_start = s->cr[VGA_CRTC_CURSOR_START];

            s->cursor_end = s->cr[VGA_CRTC_CURSOR_END];

        }



        src = (uint32_t *) s->vram_ptr + s->start_addr;

        dst = chardata;



        if (full_update) {

            for (i = 0; i < size; src ++, dst ++, i ++)

                console_write_ch(dst, VMEM2CHTYPE(le32_to_cpu(*src)));



            dpy_text_update(s->con, 0, 0, width, height);

        } else {

            c_max = 0;



            for (i = 0; i < size; src ++, dst ++, i ++) {

                console_write_ch(&val, VMEM2CHTYPE(le32_to_cpu(*src)));

                if (*dst != val) {

                    *dst = val;

                    c_max = i;

                    break;

                }

            }

            c_min = i;

            for (; i < size; src ++, dst ++, i ++) {

                console_write_ch(&val, VMEM2CHTYPE(le32_to_cpu(*src)));

                if (*dst != val) {

                    *dst = val;

                    c_max = i;

                }

            }



            if (c_min <= c_max) {

                i = TEXTMODE_Y(c_min);

                dpy_text_update(s->con, 0, i, width, TEXTMODE_Y(c_max) - i + 1);

            }

        }



        return;

    case GMODE_GRAPH:

        if (!full_update)

            return;



        s->get_resolution(s, &width, &height);

        snprintf(msg_buffer, sizeof(msg_buffer), "%i x %i Graphic mode",

                 width, height);

        break;

    case GMODE_BLANK:

    default:

        if (!full_update)

            return;



        snprintf(msg_buffer, sizeof(msg_buffer), "VGA Blank mode");

        break;

    }



    /* Display a message */

    s->last_width = 60;

    s->last_height = height = 3;

    dpy_text_cursor(s->con, -1, -1);

    dpy_text_resize(s->con, s->last_width, height);



    for (dst = chardata, i = 0; i < s->last_width * height; i ++)

        console_write_ch(dst ++, ' ');



    size = strlen(msg_buffer);

    width = (s->last_width - size) / 2;

    dst = chardata + s->last_width + width;

    for (i = 0; i < size; i ++)

        console_write_ch(dst ++, 0x00200100 | msg_buffer[i]);



    dpy_text_update(s->con, 0, 0, s->last_width, height);

}
