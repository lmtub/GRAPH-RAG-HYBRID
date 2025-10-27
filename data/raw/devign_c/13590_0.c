static void coroutine_fn bdrv_rw_co_entry(void *opaque)

{

    RwCo *rwco = opaque;



    if (!rwco->is_write) {

        rwco->ret = bdrv_co_do_preadv(rwco->bs, rwco->offset,

                                      rwco->qiov->size, rwco->qiov,

                                      rwco->flags);

    } else {

        rwco->ret = bdrv_co_do_pwritev(rwco->bs, rwco->offset,

                                       rwco->qiov->size, rwco->qiov,

                                       rwco->flags);

    }

}
