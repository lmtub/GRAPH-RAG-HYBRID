int bdrv_all_find_snapshot(const char *name, BlockDriverState **first_bad_bs)

{

    QEMUSnapshotInfo sn;

    int err = 0;

    BlockDriverState *bs;

    BdrvNextIterator *it = NULL;



    while (err == 0 && (it = bdrv_next(it, &bs))) {

        AioContext *ctx = bdrv_get_aio_context(bs);



        aio_context_acquire(ctx);

        if (bdrv_can_snapshot(bs)) {

            err = bdrv_snapshot_find(bs, &sn, name);

        }

        aio_context_release(ctx);

    }



    *first_bad_bs = bs;

    return err;

}
