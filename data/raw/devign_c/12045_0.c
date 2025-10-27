static void trim_aio_cancel(BlockAIOCB *acb)

{

    TrimAIOCB *iocb = container_of(acb, TrimAIOCB, common);



    /* Exit the loop so ide_issue_trim_cb will not continue  */

    iocb->j = iocb->qiov->niov - 1;

    iocb->i = (iocb->qiov->iov[iocb->j].iov_len / 8) - 1;



    iocb->ret = -ECANCELED;



    if (iocb->aiocb) {

        bdrv_aio_cancel_async(iocb->aiocb);

        iocb->aiocb = NULL;

    }

}
