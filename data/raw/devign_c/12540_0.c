void bdrv_aio_cancel(BlockDriverAIOCB *acb)

{

    if (acb->cb == bdrv_aio_rw_vector_cb) {

        VectorTranslationState *s = acb->opaque;

        acb = s->aiocb;

    }



    acb->pool->cancel(acb);

}
