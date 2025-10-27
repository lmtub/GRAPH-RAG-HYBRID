static QEMUCursor *qxl_cursor(PCIQXLDevice *qxl, QXLCursor *cursor)

{

    QEMUCursor *c;

    uint8_t *image, *mask;

    size_t size;



    c = cursor_alloc(cursor->header.width, cursor->header.height);

    c->hot_x = cursor->header.hot_spot_x;

    c->hot_y = cursor->header.hot_spot_y;

    switch (cursor->header.type) {

    case SPICE_CURSOR_TYPE_ALPHA:

        size = sizeof(uint32_t) * cursor->header.width * cursor->header.height;

        memcpy(c->data, cursor->chunk.data, size);

        if (qxl->debug > 2) {

            cursor_print_ascii_art(c, "qxl/alpha");

        }

        break;

    case SPICE_CURSOR_TYPE_MONO:

        mask  = cursor->chunk.data;

        image = mask + cursor_get_mono_bpl(c) * c->width;

        cursor_set_mono(c, 0xffffff, 0x000000, image, 1, mask);

        if (qxl->debug > 2) {

            cursor_print_ascii_art(c, "qxl/mono");

        }

        break;

    default:

        fprintf(stderr, "%s: not implemented: type %d\n",

                __FUNCTION__, cursor->header.type);

        goto fail;

    }

    return c;



fail:

    cursor_put(c);

    return NULL;

}
