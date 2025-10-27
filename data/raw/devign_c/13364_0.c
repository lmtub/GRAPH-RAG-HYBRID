int qed_read_l2_table_sync(BDRVQEDState *s, QEDRequest *request, uint64_t offset)

{

    int ret = -EINPROGRESS;



    qed_read_l2_table(s, request, offset, qed_sync_cb, &ret);

    while (ret == -EINPROGRESS) {

        aio_poll(bdrv_get_aio_context(s->bs), true);

    }



    return ret;

}
