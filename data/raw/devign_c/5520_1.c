void bdrv_commit_all(void)

{

    BlockDriverState *bs;



    QTAILQ_FOREACH(bs, &bdrv_states, list) {

        bdrv_commit(bs);

    }

}
