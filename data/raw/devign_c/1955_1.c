static BlockDriverState *get_bs_snapshots(void)

{

    BlockDriverState *bs;

    DriveInfo *dinfo;



    if (bs_snapshots)

        return bs_snapshots;

    QTAILQ_FOREACH(dinfo, &drives, next) {

        bs = dinfo->bdrv;

        if (bdrv_can_snapshot(bs))

            goto ok;

    }

    return NULL;

 ok:

    bs_snapshots = bs;

    return bs;

}
