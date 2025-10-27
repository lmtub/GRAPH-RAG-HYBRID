static int qed_write_table(BDRVQEDState *s, uint64_t offset, QEDTable *table,

                           unsigned int index, unsigned int n, bool flush)

{

    unsigned int sector_mask = BDRV_SECTOR_SIZE / sizeof(uint64_t) - 1;

    unsigned int start, end, i;

    QEDTable *new_table;

    struct iovec iov;

    QEMUIOVector qiov;

    size_t len_bytes;

    int ret;



    trace_qed_write_table(s, offset, table, index, n);



    /* Calculate indices of the first and one after last elements */

    start = index & ~sector_mask;

    end = (index + n + sector_mask) & ~sector_mask;



    len_bytes = (end - start) * sizeof(uint64_t);



    new_table = qemu_blockalign(s->bs, len_bytes);

    iov = (struct iovec) {

        .iov_base = new_table->offsets,

        .iov_len = len_bytes,

    };

    qemu_iovec_init_external(&qiov, &iov, 1);



    /* Byteswap table */

    for (i = start; i < end; i++) {

        uint64_t le_offset = cpu_to_le64(table->offsets[i]);

        new_table->offsets[i - start] = le_offset;

    }



    /* Adjust for offset into table */

    offset += start * sizeof(uint64_t);



    ret = bdrv_pwritev(s->bs->file, offset, &qiov);

    trace_qed_write_table_cb(s, table, flush, ret);

    if (ret < 0) {

        goto out;

    }



    if (flush) {

        qed_acquire(s);

        ret = bdrv_flush(s->bs);

        qed_release(s);

        if (ret < 0) {

            goto out;

        }

    }



    ret = 0;

out:

    qemu_vfree(new_table);

    return ret;

}
