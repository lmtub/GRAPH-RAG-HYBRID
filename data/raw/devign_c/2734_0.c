static void multiwrite_cb(void *opaque, int ret)

{

    MultiwriteCB *mcb = opaque;



    trace_multiwrite_cb(mcb, ret);



    if (ret < 0 && !mcb->error) {

        mcb->error = ret;

    }



    mcb->num_requests--;

    if (mcb->num_requests == 0) {

        multiwrite_user_cb(mcb);

        g_free(mcb);

    }

}
