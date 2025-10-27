static QEMUCursor *qxl_cursor(PCIQXLDevice *qxl, QXLCursor *cursor,

                              uint32_t group_id)

{

    QEMUCursor *c;

    size_t size;



    c = cursor_alloc(cursor->header.width, cursor->header.height);

    c->hot_x = cursor->header.hot_spot_x;

    c->hot_y = cursor->header.hot_spot_y;

    switch (cursor->header.type) {

    case SPICE_CURSOR_TYPE_ALPHA:

        size = sizeof(uint32_t) * cursor->header.width * cursor->header.height;

        qxl_unpack_chunks(c->data, size, qxl, &cursor->chunk, group_id);

        if (qxl->debug > 2) {

            cursor_print_ascii_art(c, "qxl/alpha");

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
