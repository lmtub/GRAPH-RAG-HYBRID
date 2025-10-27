static void blkverify_aio_cb(void *opaque, int ret)

{

    BlkverifyAIOCB *acb = opaque;



    switch (++acb->done) {

    case 1:

        acb->ret = ret;

        break;



    case 2:

        if (acb->ret != ret) {

            blkverify_err(acb, "return value mismatch %d != %d", acb->ret, ret);

        }



        if (acb->verify) {

            acb->verify(acb);

        }



        aio_bh_schedule_oneshot(bdrv_get_aio_context(acb->common.bs),

                                blkverify_aio_bh, acb);

        break;

    }

}
