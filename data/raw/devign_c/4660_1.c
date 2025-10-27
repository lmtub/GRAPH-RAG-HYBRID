static void vnc_update_client(void *opaque)

{

    VncState *vs = opaque;

    if (vs->need_update && vs->csock != -1) {

        int y;

        uint8_t *row;

        char *old_row;

        uint32_t width_mask[VNC_DIRTY_WORDS];

        int n_rectangles;

        int saved_offset;

        int has_dirty = 0;



        vga_hw_update();



        vnc_set_bits(width_mask, (ds_get_width(vs->ds) / 16), VNC_DIRTY_WORDS);



        /* Walk through the dirty map and eliminate tiles that

           really aren't dirty */

        row = ds_get_data(vs->ds);

        old_row = vs->old_data;



        for (y = 0; y < ds_get_height(vs->ds); y++) {

            if (vnc_and_bits(vs->dirty_row[y], width_mask, VNC_DIRTY_WORDS)) {

                int x;

                uint8_t *ptr;

                char *old_ptr;



                ptr = row;

                old_ptr = (char*)old_row;



                for (x = 0; x < ds_get_width(vs->ds); x += 16) {

                    if (memcmp(old_ptr, ptr, 16 * ds_get_bytes_per_pixel(vs->ds)) == 0) {

                        vnc_clear_bit(vs->dirty_row[y], (x / 16));

                    } else {

                        has_dirty = 1;

                        memcpy(old_ptr, ptr, 16 * ds_get_bytes_per_pixel(vs->ds));

                    }



                    ptr += 16 * ds_get_bytes_per_pixel(vs->ds);

                    old_ptr += 16 * ds_get_bytes_per_pixel(vs->ds);

                }

            }



            row += ds_get_linesize(vs->ds);

            old_row += ds_get_linesize(vs->ds);

        }



        if (!has_dirty && !vs->audio_cap) {

            qemu_mod_timer(vs->timer, qemu_get_clock(rt_clock) + VNC_REFRESH_INTERVAL);

            return;

        }



        /* Count rectangles */

        n_rectangles = 0;

        vnc_write_u8(vs, 0);  /* msg id */

        vnc_write_u8(vs, 0);

        saved_offset = vs->output.offset;

        vnc_write_u16(vs, 0);



        for (y = 0; y < vs->serverds.height; y++) {

            int x;

            int last_x = -1;

            for (x = 0; x < vs->serverds.width / 16; x++) {

                if (vnc_get_bit(vs->dirty_row[y], x)) {

                    if (last_x == -1) {

                        last_x = x;

                    }

                    vnc_clear_bit(vs->dirty_row[y], x);

                } else {

                    if (last_x != -1) {

                        int h = find_dirty_height(vs, y, last_x, x);

                        send_framebuffer_update(vs, last_x * 16, y, (x - last_x) * 16, h);

                        n_rectangles++;

                    }

                    last_x = -1;

                }

            }

            if (last_x != -1) {

                int h = find_dirty_height(vs, y, last_x, x);

                send_framebuffer_update(vs, last_x * 16, y, (x - last_x) * 16, h);

                n_rectangles++;

            }

        }

        vs->output.buffer[saved_offset] = (n_rectangles >> 8) & 0xFF;

        vs->output.buffer[saved_offset + 1] = n_rectangles & 0xFF;

        vnc_flush(vs);



    }



    if (vs->csock != -1) {

        qemu_mod_timer(vs->timer, qemu_get_clock(rt_clock) + VNC_REFRESH_INTERVAL);

    }



}
