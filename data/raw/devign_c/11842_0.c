SimpleSpiceUpdate *qemu_spice_create_update(SimpleSpiceDisplay *ssd)

{

    SimpleSpiceUpdate *update;

    QXLDrawable *drawable;

    QXLImage *image;

    QXLCommand *cmd;

    uint8_t *src, *dst;

    int by, bw, bh;



    if (qemu_spice_rect_is_empty(&ssd->dirty)) {

        return NULL;

    };



    pthread_mutex_lock(&ssd->lock);

    dprint(2, "%s: lr %d -> %d,  tb -> %d -> %d\n", __FUNCTION__,

           ssd->dirty.left, ssd->dirty.right,

           ssd->dirty.top, ssd->dirty.bottom);



    update   = qemu_mallocz(sizeof(*update));

    drawable = &update->drawable;

    image    = &update->image;

    cmd      = &update->ext.cmd;



    bw       = ssd->dirty.right - ssd->dirty.left;

    bh       = ssd->dirty.bottom - ssd->dirty.top;

    update->bitmap = qemu_malloc(bw * bh * 4);



    drawable->bbox            = ssd->dirty;

    drawable->clip.type       = SPICE_CLIP_TYPE_NONE;

    drawable->effect          = QXL_EFFECT_OPAQUE;

    drawable->release_info.id = (intptr_t)update;

    drawable->type            = QXL_DRAW_COPY;

    drawable->surfaces_dest[0] = -1;

    drawable->surfaces_dest[1] = -1;

    drawable->surfaces_dest[2] = -1;



    drawable->u.copy.rop_descriptor  = SPICE_ROPD_OP_PUT;

    drawable->u.copy.src_bitmap      = (intptr_t)image;

    drawable->u.copy.src_area.right  = bw;

    drawable->u.copy.src_area.bottom = bh;



    QXL_SET_IMAGE_ID(image, QXL_IMAGE_GROUP_DEVICE, ssd->unique++);

    image->descriptor.type   = SPICE_IMAGE_TYPE_BITMAP;

    image->bitmap.flags      = QXL_BITMAP_DIRECT | QXL_BITMAP_TOP_DOWN;

    image->bitmap.stride     = bw * 4;

    image->descriptor.width  = image->bitmap.x = bw;

    image->descriptor.height = image->bitmap.y = bh;

    image->bitmap.data = (intptr_t)(update->bitmap);

    image->bitmap.palette = 0;

    image->bitmap.format = SPICE_BITMAP_FMT_32BIT;



    if (ssd->conv == NULL) {

        PixelFormat dst = qemu_default_pixelformat(32);

        ssd->conv = qemu_pf_conv_get(&dst, &ssd->ds->surface->pf);

        assert(ssd->conv);

    }



    src = ds_get_data(ssd->ds) +

        ssd->dirty.top * ds_get_linesize(ssd->ds) +

        ssd->dirty.left * ds_get_bytes_per_pixel(ssd->ds);

    dst = update->bitmap;

    for (by = 0; by < bh; by++) {

        qemu_pf_conv_run(ssd->conv, dst, src, bw);

        src += ds_get_linesize(ssd->ds);

        dst += image->bitmap.stride;

    }



    cmd->type = QXL_CMD_DRAW;

    cmd->data = (intptr_t)drawable;



    memset(&ssd->dirty, 0, sizeof(ssd->dirty));

    pthread_mutex_unlock(&ssd->lock);

    return update;

}
