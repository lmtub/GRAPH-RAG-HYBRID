int qed_read_l1_table_sync(BDRVQEDState *s)

{

    int ret = -EINPROGRESS;



    async_context_push();



    qed_read_table(s, s->header.l1_table_offset,

                   s->l1_table, qed_sync_cb, &ret);

    while (ret == -EINPROGRESS) {

        qemu_aio_wait();

    }



    async_context_pop();



    return ret;

}
