static int qed_read_table(BDRVQEDState *s, uint64_t offset, QEDTable *table)

{

    QEMUIOVector qiov;

    int noffsets;

    int i, ret;



    struct iovec iov = {

        .iov_base = table->offsets,

        .iov_len = s->header.cluster_size * s->header.table_size,

    };

    qemu_iovec_init_external(&qiov, &iov, 1);



    trace_qed_read_table(s, offset, table);



    ret = bdrv_preadv(s->bs->file, offset, &qiov);

    if (ret < 0) {

        goto out;

    }



    /* Byteswap offsets */

    qed_acquire(s);

    noffsets = qiov.size / sizeof(uint64_t);

    for (i = 0; i < noffsets; i++) {

        table->offsets[i] = le64_to_cpu(table->offsets[i]);

    }

    qed_release(s);



    ret = 0;

out:

    /* Completion */

    trace_qed_read_table_cb(s, table, ret);

    return ret;

}
