void bdrv_aio_cancel_async(BlockAIOCB *acb)

{

    if (acb->aiocb_info->cancel_async) {

        acb->aiocb_info->cancel_async(acb);

    }

}
