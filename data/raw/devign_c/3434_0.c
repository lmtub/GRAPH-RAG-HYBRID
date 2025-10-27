static int send_png_rect(VncState *vs, int x, int y, int w, int h,

                         QDict *palette)

{

    png_byte color_type;

    png_structp png_ptr;

    png_infop info_ptr;

    png_colorp png_palette = NULL;

    size_t offset;

    int level = tight_conf[vs->tight_compression].raw_zlib_level;

    uint8_t *buf;

    int dy;



    png_ptr = png_create_write_struct_2(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL,

                                        NULL, vnc_png_malloc, vnc_png_free);



    if (png_ptr == NULL)

        return -1;



    info_ptr = png_create_info_struct(png_ptr);



    if (info_ptr == NULL) {

        png_destroy_write_struct(&png_ptr, NULL);

        return -1;

    }



    png_set_write_fn(png_ptr, (void *) vs, png_write_data, png_flush_data);

    png_set_compression_level(png_ptr, level);



    if (palette) {

        color_type = PNG_COLOR_TYPE_PALETTE;

    } else {

        color_type = PNG_COLOR_TYPE_RGB;

    }



    png_set_IHDR(png_ptr, info_ptr, w, h,

                 8, color_type, PNG_INTERLACE_NONE,

                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);



    if (color_type == PNG_COLOR_TYPE_PALETTE) {

        struct palette_cb_priv priv;



        png_palette = png_malloc(png_ptr, sizeof(*png_palette) *

                                 qdict_size(palette));



        priv.vs = vs;

        priv.png_palette = png_palette;

        qdict_iter(palette, write_png_palette, &priv);



        png_set_PLTE(png_ptr, info_ptr, png_palette, qdict_size(palette));



        offset = vs->tight.offset;

        if (vs->clientds.pf.bytes_per_pixel == 4) {

            tight_encode_indexed_rect32(vs->tight.buffer, w * h, palette);

        } else {

            tight_encode_indexed_rect16(vs->tight.buffer, w * h, palette);

        }

    }



    png_write_info(png_ptr, info_ptr);



    buffer_reserve(&vs->tight_png, 2048);

    buf = qemu_malloc(w * 3);

    for (dy = 0; dy < h; dy++)

    {

        if (color_type == PNG_COLOR_TYPE_PALETTE) {

            memcpy(buf, vs->tight.buffer + (dy * w), w);

        } else {

            rgb_prepare_row(vs, buf, x, y + dy, w);

        }

        png_write_row(png_ptr, buf);

    }

    qemu_free(buf);



    png_write_end(png_ptr, NULL);



    if (color_type == PNG_COLOR_TYPE_PALETTE) {

        png_free(png_ptr, png_palette);

    }



    png_destroy_write_struct(&png_ptr, &info_ptr);



    vnc_write_u8(vs, VNC_TIGHT_PNG << 4);



    tight_send_compact_size(vs, vs->tight_png.offset);

    vnc_write(vs, vs->tight_png.buffer, vs->tight_png.offset);

    buffer_reset(&vs->tight_png);

    return 1;

}
