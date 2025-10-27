int bdrv_aio_multiwrite(BlockDriverState *bs, BlockRequest *reqs, int num_reqs)

{

    BlockDriverAIOCB *acb;

    MultiwriteCB *mcb;

    int i;



    if (num_reqs == 0) {

        return 0;

    }



    // Create MultiwriteCB structure

    mcb = qemu_mallocz(sizeof(*mcb) + num_reqs * sizeof(*mcb->callbacks));

    mcb->num_requests = 0;

    mcb->num_callbacks = num_reqs;



    for (i = 0; i < num_reqs; i++) {

        mcb->callbacks[i].cb = reqs[i].cb;

        mcb->callbacks[i].opaque = reqs[i].opaque;

    }



    // Check for mergable requests

    num_reqs = multiwrite_merge(bs, reqs, num_reqs, mcb);



    // Run the aio requests

    for (i = 0; i < num_reqs; i++) {

        acb = bdrv_aio_writev(bs, reqs[i].sector, reqs[i].qiov,

            reqs[i].nb_sectors, multiwrite_cb, mcb);



        if (acb == NULL) {

            // We can only fail the whole thing if no request has been

            // submitted yet. Otherwise we'll wait for the submitted AIOs to

            // complete and report the error in the callback.

            if (mcb->num_requests == 0) {

                reqs[i].error = -EIO;

                goto fail;

            } else {

                mcb->error = -EIO;

                break;

            }

        } else {

            mcb->num_requests++;

        }

    }



    return 0;



fail:

    free(mcb);

    return -1;

}
