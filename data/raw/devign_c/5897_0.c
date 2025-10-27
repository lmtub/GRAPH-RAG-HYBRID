void qmp_block_dirty_bitmap_clear(const char *node, const char *name,

                                  Error **errp)

{

    AioContext *aio_context;

    BdrvDirtyBitmap *bitmap;

    BlockDriverState *bs;



    bitmap = block_dirty_bitmap_lookup(node, name, &bs, &aio_context, errp);

    if (!bitmap || !bs) {

        return;

    }



    if (bdrv_dirty_bitmap_frozen(bitmap)) {

        error_setg(errp,

                   "Bitmap '%s' is currently frozen and cannot be modified",

                   name);

        goto out;

    } else if (!bdrv_dirty_bitmap_enabled(bitmap)) {

        error_setg(errp,

                   "Bitmap '%s' is currently disabled and cannot be cleared",

                   name);

        goto out;

    }



    bdrv_clear_dirty_bitmap(bitmap, NULL);



 out:

    aio_context_release(aio_context);

}
