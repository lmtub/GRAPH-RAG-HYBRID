static int qcow2_mark_clean(BlockDriverState *bs)

{

    BDRVQcowState *s = bs->opaque;



    if (s->incompatible_features & QCOW2_INCOMPAT_DIRTY) {

        int ret = bdrv_flush(bs);

        if (ret < 0) {

            return ret;

        }



        s->incompatible_features &= ~QCOW2_INCOMPAT_DIRTY;

        return qcow2_update_header(bs);

    }

    return 0;

}
