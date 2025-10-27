static void bdrv_close(BlockDriverState *bs)

{

    BdrvAioNotifier *ban, *ban_next;



    assert(!bs->job);

    assert(!bs->refcnt);



    bdrv_drained_begin(bs); /* complete I/O */

    bdrv_flush(bs);

    bdrv_drain(bs); /* in case flush left pending I/O */



    if (bs->drv) {

        BdrvChild *child, *next;



        bs->drv->bdrv_close(bs);

        bs->drv = NULL;



        bdrv_set_backing_hd(bs, NULL, &error_abort);



        if (bs->file != NULL) {

            bdrv_unref_child(bs, bs->file);

            bs->file = NULL;

        }



        QLIST_FOREACH_SAFE(child, &bs->children, next, next) {

            /* TODO Remove bdrv_unref() from drivers' close function and use

             * bdrv_unref_child() here */

            if (child->bs->inherits_from == bs) {

                child->bs->inherits_from = NULL;

            }

            bdrv_detach_child(child);

        }



        g_free(bs->opaque);

        bs->opaque = NULL;

        atomic_set(&bs->copy_on_read, 0);

        bs->backing_file[0] = '\0';

        bs->backing_format[0] = '\0';

        bs->total_sectors = 0;

        bs->encrypted = false;

        bs->sg = false;

        QDECREF(bs->options);

        QDECREF(bs->explicit_options);

        bs->options = NULL;

        bs->explicit_options = NULL;

        QDECREF(bs->full_open_options);

        bs->full_open_options = NULL;

    }



    bdrv_release_named_dirty_bitmaps(bs);

    assert(QLIST_EMPTY(&bs->dirty_bitmaps));



    QLIST_FOREACH_SAFE(ban, &bs->aio_notifiers, list, ban_next) {

        g_free(ban);

    }

    QLIST_INIT(&bs->aio_notifiers);

    bdrv_drained_end(bs);

}
