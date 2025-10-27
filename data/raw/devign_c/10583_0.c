static BdrvDirtyBitmap *block_dirty_bitmap_lookup(const char *node,

                                                  const char *name,

                                                  BlockDriverState **pbs,

                                                  AioContext **paio,

                                                  Error **errp)

{

    BlockDriverState *bs;

    BdrvDirtyBitmap *bitmap;

    AioContext *aio_context;



    if (!node) {

        error_setg(errp, "Node cannot be NULL");

        return NULL;

    }

    if (!name) {

        error_setg(errp, "Bitmap name cannot be NULL");

        return NULL;

    }

    bs = bdrv_lookup_bs(node, node, NULL);

    if (!bs) {

        error_setg(errp, "Node '%s' not found", node);

        return NULL;

    }



    aio_context = bdrv_get_aio_context(bs);

    aio_context_acquire(aio_context);



    bitmap = bdrv_find_dirty_bitmap(bs, name);

    if (!bitmap) {

        error_setg(errp, "Dirty bitmap '%s' not found", name);

        goto fail;

    }



    if (pbs) {

        *pbs = bs;

    }

    if (paio) {

        *paio = aio_context;

    } else {

        aio_context_release(aio_context);

    }



    return bitmap;



 fail:

    aio_context_release(aio_context);

    return NULL;

}
