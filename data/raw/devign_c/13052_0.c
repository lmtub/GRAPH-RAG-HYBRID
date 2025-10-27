void qmp_block_dirty_bitmap_add(const char *node, const char *name,

                                bool has_granularity, uint32_t granularity,

                                Error **errp)

{

    AioContext *aio_context;

    BlockDriverState *bs;



    if (!name || name[0] == '\0') {

        error_setg(errp, "Bitmap name cannot be empty");

        return;

    }



    bs = bdrv_lookup_bs(node, node, errp);

    if (!bs) {

        return;

    }



    aio_context = bdrv_get_aio_context(bs);

    aio_context_acquire(aio_context);



    if (has_granularity) {

        if (granularity < 512 || !is_power_of_2(granularity)) {

            error_setg(errp, "Granularity must be power of 2 "

                             "and at least 512");

            goto out;

        }

    } else {

        /* Default to cluster size, if available: */

        granularity = bdrv_get_default_bitmap_granularity(bs);

    }



    bdrv_create_dirty_bitmap(bs, granularity, name, errp);



 out:

    aio_context_release(aio_context);

}
