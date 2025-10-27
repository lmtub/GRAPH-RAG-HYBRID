void virtio_submit_multiwrite(BlockDriverState *bs, MultiReqBuffer *mrb)

{

    int i, ret;



    if (!mrb->num_writes) {

        return;

    }



    ret = bdrv_aio_multiwrite(bs, mrb->blkreq, mrb->num_writes);

    if (ret != 0) {

        for (i = 0; i < mrb->num_writes; i++) {

            if (mrb->blkreq[i].error) {

                virtio_blk_rw_complete(mrb->blkreq[i].opaque, -EIO);

            }

        }

    }



    mrb->num_writes = 0;

}
