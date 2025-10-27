static void multiwrite_user_cb(MultiwriteCB *mcb)

{

    int i;



    for (i = 0; i < mcb->num_callbacks; i++) {

        mcb->callbacks[i].cb(mcb->callbacks[i].opaque, mcb->error);

        qemu_free(mcb->callbacks[i].free_qiov);

        qemu_free(mcb->callbacks[i].free_buf);

    }

}
