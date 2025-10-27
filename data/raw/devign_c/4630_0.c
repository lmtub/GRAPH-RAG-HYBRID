void destroy_bdrvs(dev_match_fn *match_fn, void *arg)

{

    DriveInfo *dinfo;

    struct BlockDriverState *bs;



    TAILQ_FOREACH(dinfo, &drives, next) {

        bs = dinfo->bdrv;

        if (bs) {

            if (bs->private && match_fn(bs->private, arg)) {

                drive_uninit(bs);

                bdrv_delete(bs);

            }

        }

    }

}
