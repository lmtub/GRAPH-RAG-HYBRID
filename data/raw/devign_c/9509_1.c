static void vga_draw_graphic(VGAState *s, int full_update)

{

    int y1, y, update, page_min, page_max, linesize, y_start, double_scan, mask, depth;

    int width, height, shift_control, line_offset, page0, page1, bwidth, bits;

    int disp_width, multi_scan, multi_run;

    uint8_t *d;

    uint32_t v, addr1, addr;

    vga_draw_line_func *vga_draw_line;



    full_update |= update_basic_params(s);



    if (!full_update)

        vga_sync_dirty_bitmap(s);



    s->get_resolution(s, &width, &height);

    disp_width = width;



    shift_control = (s->gr[0x05] >> 5) & 3;

    double_scan = (s->cr[0x09] >> 7);

    if (shift_control != 1) {

        multi_scan = (((s->cr[0x09] & 0x1f) + 1) << double_scan) - 1;

    } else {

        /* in CGA modes, multi_scan is ignored */

        /* XXX: is it correct ? */

        multi_scan = double_scan;

    }

    multi_run = multi_scan;

    if (shift_control != s->shift_control ||

        double_scan != s->double_scan) {

        full_update = 1;

        s->shift_control = shift_control;

        s->double_scan = double_scan;

    }



    if (shift_control == 0) {

        full_update |= update_palette16(s);

        if (s->sr[0x01] & 8) {

            v = VGA_DRAW_LINE4D2;

            disp_width <<= 1;

        } else {

            v = VGA_DRAW_LINE4;

        }

        bits = 4;

    } else if (shift_control == 1) {

        full_update |= update_palette16(s);

        if (s->sr[0x01] & 8) {

            v = VGA_DRAW_LINE2D2;

            disp_width <<= 1;

        } else {

            v = VGA_DRAW_LINE2;

        }

        bits = 4;

    } else {

        switch(s->get_bpp(s)) {

        default:

        case 0:

            full_update |= update_palette256(s);

            v = VGA_DRAW_LINE8D2;

            bits = 4;

            break;

        case 8:

            full_update |= update_palette256(s);

            v = VGA_DRAW_LINE8;

            bits = 8;

            break;

        case 15:

            v = VGA_DRAW_LINE15;

            bits = 16;

            break;

        case 16:

            v = VGA_DRAW_LINE16;

            bits = 16;

            break;

        case 24:

            v = VGA_DRAW_LINE24;

            bits = 24;

            break;

        case 32:

            v = VGA_DRAW_LINE32;

            bits = 32;

            break;

        }

    }

    vga_draw_line = vga_draw_line_table[v * NB_DEPTHS + get_depth_index(s->ds)];



    depth = s->get_bpp(s);

    if (s->line_offset != s->last_line_offset ||

        disp_width != s->last_width ||

        height != s->last_height ||

        s->last_depth != depth) {

        if (depth == 16 || depth == 32) {

            if (is_graphic_console()) {

                qemu_free_displaysurface(s->ds->surface);

                s->ds->surface = qemu_create_displaysurface_from(disp_width, height, depth,

                                                               s->line_offset,

                                                               s->vram_ptr + (s->start_addr * 4));

                dpy_resize(s->ds);

            } else {

                qemu_console_resize(s->ds, disp_width, height);

            }

        } else {

            qemu_console_resize(s->ds, disp_width, height);

        }

        s->last_scr_width = disp_width;

        s->last_scr_height = height;

        s->last_width = disp_width;

        s->last_height = height;

        s->last_line_offset = s->line_offset;

        s->last_depth = depth;

        full_update = 1;

    } else if (is_graphic_console() && is_buffer_shared(s->ds->surface) &&

               (full_update || s->ds->surface->data != s->vram_ptr + (s->start_addr * 4))) {

        s->ds->surface->data = s->vram_ptr + (s->start_addr * 4);

        dpy_setdata(s->ds);

    }



    s->rgb_to_pixel =

        rgb_to_pixel_dup_table[get_depth_index(s->ds)];



    if (!is_buffer_shared(s->ds->surface) && s->cursor_invalidate)

        s->cursor_invalidate(s);



    line_offset = s->line_offset;

#if 0

    printf("w=%d h=%d v=%d line_offset=%d cr[0x09]=0x%02x cr[0x17]=0x%02x linecmp=%d sr[0x01]=0x%02x\n",

           width, height, v, line_offset, s->cr[9], s->cr[0x17], s->line_compare, s->sr[0x01]);

#endif

    addr1 = (s->start_addr * 4);

    bwidth = (width * bits + 7) / 8;

    y_start = -1;

    page_min = 0x7fffffff;

    page_max = -1;

    d = ds_get_data(s->ds);

    linesize = ds_get_linesize(s->ds);

    y1 = 0;

    for(y = 0; y < height; y++) {

        addr = addr1;

        if (!(s->cr[0x17] & 1)) {

            int shift;

            /* CGA compatibility handling */

            shift = 14 + ((s->cr[0x17] >> 6) & 1);

            addr = (addr & ~(1 << shift)) | ((y1 & 1) << shift);

        }

        if (!(s->cr[0x17] & 2)) {

            addr = (addr & ~0x8000) | ((y1 & 2) << 14);

        }

        page0 = s->vram_offset + (addr & TARGET_PAGE_MASK);

        page1 = s->vram_offset + ((addr + bwidth - 1) & TARGET_PAGE_MASK);

        update = full_update |

            cpu_physical_memory_get_dirty(page0, VGA_DIRTY_FLAG) |

            cpu_physical_memory_get_dirty(page1, VGA_DIRTY_FLAG);

        if ((page1 - page0) > TARGET_PAGE_SIZE) {

            /* if wide line, can use another page */

            update |= cpu_physical_memory_get_dirty(page0 + TARGET_PAGE_SIZE,

                                                    VGA_DIRTY_FLAG);

        }

        /* explicit invalidation for the hardware cursor */

        update |= (s->invalidated_y_table[y >> 5] >> (y & 0x1f)) & 1;

        if (update) {

            if (y_start < 0)

                y_start = y;

            if (page0 < page_min)

                page_min = page0;

            if (page1 > page_max)

                page_max = page1;

            if (!(is_buffer_shared(s->ds->surface))) {

                vga_draw_line(s, d, s->vram_ptr + addr, width);

                if (s->cursor_draw_line)

                    s->cursor_draw_line(s, d, y);

            }

        } else {

            if (y_start >= 0) {

                /* flush to display */

                dpy_update(s->ds, 0, y_start,

                           disp_width, y - y_start);

                y_start = -1;

            }

        }

        if (!multi_run) {

            mask = (s->cr[0x17] & 3) ^ 3;

            if ((y1 & mask) == mask)

                addr1 += line_offset;

            y1++;

            multi_run = multi_scan;

        } else {

            multi_run--;

        }

        /* line compare acts on the displayed lines */

        if (y == s->line_compare)

            addr1 = 0;

        d += linesize;

    }

    if (y_start >= 0) {

        /* flush to display */

        dpy_update(s->ds, 0, y_start,

                   disp_width, y - y_start);

    }

    /* reset modified pages */

    if (page_max != -1) {

        cpu_physical_memory_reset_dirty(page_min, page_max + TARGET_PAGE_SIZE,

                                        VGA_DIRTY_FLAG);

    }

    memset(s->invalidated_y_table, 0, ((height + 31) >> 5) * 4);

}
