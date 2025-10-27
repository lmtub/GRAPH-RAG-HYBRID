static void vnc_colordepth(DisplayState *ds)

{

    int host_big_endian_flag;

    struct VncState *vs = ds->opaque;



#ifdef WORDS_BIGENDIAN

    host_big_endian_flag = 1;

#else

    host_big_endian_flag = 0;

#endif   

    

    switch (ds_get_bits_per_pixel(ds)) {

        case 8:

            vs->depth = 1;

            vs->server_red_max = 7;

            vs->server_green_max = 7;

            vs->server_blue_max = 3;

            vs->server_red_shift = 5;

            vs->server_green_shift = 2;

            vs->server_blue_shift = 0;

            break;

        case 16:

            vs->depth = 2;

            vs->server_red_max = 31;

            vs->server_green_max = 63;

            vs->server_blue_max = 31;

            vs->server_red_shift = 11;

            vs->server_green_shift = 5;

            vs->server_blue_shift = 0;

            break;

        case 32:

            vs->depth = 4;

            vs->server_red_max = 255;

            vs->server_green_max = 255;

            vs->server_blue_max = 255;

            vs->server_red_shift = 16;

            vs->server_green_shift = 8;

            vs->server_blue_shift = 0;

            break;

        default:

            return;

    }



    if (vs->csock != -1 && vs->has_WMVi) {

        /* Sending a WMVi message to notify the client*/

        vnc_write_u8(vs, 0);  /* msg id */

        vnc_write_u8(vs, 0);

        vnc_write_u16(vs, 1); /* number of rects */

        vnc_framebuffer_update(vs, 0, 0, ds_get_width(ds), ds_get_height(ds), 0x574D5669);

        pixel_format_message(vs);

        vnc_flush(vs);

    } else {

        if (vs->pix_bpp == 4 && vs->depth == 4 &&

                host_big_endian_flag == vs->pix_big_endian &&

                vs->client_red_max == 0xff && vs->client_green_max == 0xff && vs->client_blue_max == 0xff &&

                vs->client_red_shift == 16 && vs->client_green_shift == 8 && vs->client_blue_shift == 0) {

            vs->write_pixels = vnc_write_pixels_copy;

            vs->send_hextile_tile = send_hextile_tile_32;

        } else if (vs->pix_bpp == 2 && vs->depth == 2 &&

                host_big_endian_flag == vs->pix_big_endian &&

                vs->client_red_max == 31 && vs->client_green_max == 63 && vs->client_blue_max == 31 &&

                vs->client_red_shift == 11 && vs->client_green_shift == 5 && vs->client_blue_shift == 0) {

            vs->write_pixels = vnc_write_pixels_copy;

            vs->send_hextile_tile = send_hextile_tile_16;

        } else if (vs->pix_bpp == 1 && vs->depth == 1 &&

                host_big_endian_flag == vs->pix_big_endian &&

                vs->client_red_max == 7 && vs->client_green_max == 7 && vs->client_blue_max == 3 &&

                vs->client_red_shift == 5 && vs->client_green_shift == 2 && vs->client_blue_shift == 0) {

            vs->write_pixels = vnc_write_pixels_copy;

            vs->send_hextile_tile = send_hextile_tile_8;

        } else {

            if (vs->depth == 4) {

                vs->send_hextile_tile = send_hextile_tile_generic_32;

            } else if (vs->depth == 2) {

                vs->send_hextile_tile = send_hextile_tile_generic_16;

            } else {

                vs->send_hextile_tile = send_hextile_tile_generic_8;

            }

            vs->write_pixels = vnc_write_pixels_generic;

        }

    }

}
