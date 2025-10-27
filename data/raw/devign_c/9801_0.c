static void bdrv_qed_invalidate_cache(BlockDriverState *bs, Error **errp)

{

    BDRVQEDState *s = bs->opaque;

    Error *local_err = NULL;

    int ret;



    bdrv_qed_close(bs);



    memset(s, 0, sizeof(BDRVQEDState));

    ret = bdrv_qed_do_open(bs, NULL, bs->open_flags, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        error_prepend(errp, "Could not reopen qed layer: ");

        return;

    } else if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not reopen qed layer");

        return;

    }

}
