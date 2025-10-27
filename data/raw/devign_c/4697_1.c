static int perform_cow(BlockDriverState *bs, QCowL2Meta *m)

{

    BDRVQcow2State *s = bs->opaque;

    Qcow2COWRegion *start = &m->cow_start;

    Qcow2COWRegion *end = &m->cow_end;

    unsigned buffer_size;

    unsigned data_bytes = end->offset - (start->offset + start->nb_bytes);

    bool merge_reads;

    uint8_t *start_buffer, *end_buffer;

    QEMUIOVector qiov;

    int ret;



    assert(start->nb_bytes <= UINT_MAX - end->nb_bytes);

    assert(start->nb_bytes + end->nb_bytes <= UINT_MAX - data_bytes);

    assert(start->offset + start->nb_bytes <= end->offset);

    assert(!m->data_qiov || m->data_qiov->size == data_bytes);



    if (start->nb_bytes == 0 && end->nb_bytes == 0) {

        return 0;

    }



    /* If we have to read both the start and end COW regions and the

     * middle region is not too large then perform just one read

     * operation */

    merge_reads = start->nb_bytes && end->nb_bytes && data_bytes <= 16384;

    if (merge_reads) {

        buffer_size = start->nb_bytes + data_bytes + end->nb_bytes;

    } else {

        /* If we have to do two reads, add some padding in the middle

         * if necessary to make sure that the end region is optimally

         * aligned. */

        size_t align = bdrv_opt_mem_align(bs);

        assert(align > 0 && align <= UINT_MAX);

        assert(QEMU_ALIGN_UP(start->nb_bytes, align) <=

               UINT_MAX - end->nb_bytes);

        buffer_size = QEMU_ALIGN_UP(start->nb_bytes, align) + end->nb_bytes;

    }



    /* Reserve a buffer large enough to store all the data that we're

     * going to read */

    start_buffer = qemu_try_blockalign(bs, buffer_size);

    if (start_buffer == NULL) {

        return -ENOMEM;

    }

    /* The part of the buffer where the end region is located */

    end_buffer = start_buffer + buffer_size - end->nb_bytes;



    qemu_iovec_init(&qiov, 2 + (m->data_qiov ? m->data_qiov->niov : 0));



    qemu_co_mutex_unlock(&s->lock);

    /* First we read the existing data from both COW regions. We

     * either read the whole region in one go, or the start and end

     * regions separately. */

    if (merge_reads) {

        qemu_iovec_add(&qiov, start_buffer, buffer_size);

        ret = do_perform_cow_read(bs, m->offset, start->offset, &qiov);

    } else {

        qemu_iovec_add(&qiov, start_buffer, start->nb_bytes);

        ret = do_perform_cow_read(bs, m->offset, start->offset, &qiov);

        if (ret < 0) {

            goto fail;

        }



        qemu_iovec_reset(&qiov);

        qemu_iovec_add(&qiov, end_buffer, end->nb_bytes);

        ret = do_perform_cow_read(bs, m->offset, end->offset, &qiov);

    }

    if (ret < 0) {

        goto fail;

    }



    /* Encrypt the data if necessary before writing it */

    if (bs->encrypted) {

        if (!do_perform_cow_encrypt(bs, m->offset, start->offset,

                                    start_buffer, start->nb_bytes) ||

            !do_perform_cow_encrypt(bs, m->offset, end->offset,

                                    end_buffer, end->nb_bytes)) {

            ret = -EIO;

            goto fail;

        }

    }



    /* And now we can write everything. If we have the guest data we

     * can write everything in one single operation */

    if (m->data_qiov) {

        qemu_iovec_reset(&qiov);

        if (start->nb_bytes) {

            qemu_iovec_add(&qiov, start_buffer, start->nb_bytes);

        }

        qemu_iovec_concat(&qiov, m->data_qiov, 0, data_bytes);

        if (end->nb_bytes) {

            qemu_iovec_add(&qiov, end_buffer, end->nb_bytes);

        }

        /* NOTE: we have a write_aio blkdebug event here followed by

         * a cow_write one in do_perform_cow_write(), but there's only

         * one single I/O operation */

        BLKDBG_EVENT(bs->file, BLKDBG_WRITE_AIO);

        ret = do_perform_cow_write(bs, m->alloc_offset, start->offset, &qiov);

    } else {

        /* If there's no guest data then write both COW regions separately */

        qemu_iovec_reset(&qiov);

        qemu_iovec_add(&qiov, start_buffer, start->nb_bytes);

        ret = do_perform_cow_write(bs, m->alloc_offset, start->offset, &qiov);

        if (ret < 0) {

            goto fail;

        }



        qemu_iovec_reset(&qiov);

        qemu_iovec_add(&qiov, end_buffer, end->nb_bytes);

        ret = do_perform_cow_write(bs, m->alloc_offset, end->offset, &qiov);

    }



fail:

    qemu_co_mutex_lock(&s->lock);



    /*

     * Before we update the L2 table to actually point to the new cluster, we

     * need to be sure that the refcounts have been increased and COW was

     * handled.

     */

    if (ret == 0) {

        qcow2_cache_depends_on_flush(s->l2_table_cache);

    }



    qemu_vfree(start_buffer);

    qemu_iovec_destroy(&qiov);

    return ret;

}
