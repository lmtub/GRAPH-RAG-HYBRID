static void multiwrite_cb(void *opaque, int ret)

{

    MultiwriteCB *mcb = opaque;



    if (ret < 0) {

        mcb->error = ret;

        multiwrite_user_cb(mcb);

    }



    mcb->num_requests--;

    if (mcb->num_requests == 0) {

        if (mcb->error == 0) {

            multiwrite_user_cb(mcb);

        }

        qemu_free(mcb);

    }

}
