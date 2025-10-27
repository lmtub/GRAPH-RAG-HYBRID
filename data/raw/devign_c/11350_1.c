static void mirror_iteration_done(MirrorOp *op, int ret)

{

    MirrorBlockJob *s = op->s;

    struct iovec *iov;

    int64_t chunk_num;

    int i, nb_chunks, sectors_per_chunk;



    trace_mirror_iteration_done(s, op->sector_num, op->nb_sectors, ret);



    s->in_flight--;

    s->sectors_in_flight -= op->nb_sectors;

    iov = op->qiov.iov;

    for (i = 0; i < op->qiov.niov; i++) {

        MirrorBuffer *buf = (MirrorBuffer *) iov[i].iov_base;

        QSIMPLEQ_INSERT_TAIL(&s->buf_free, buf, next);

        s->buf_free_count++;

    }



    sectors_per_chunk = s->granularity >> BDRV_SECTOR_BITS;

    chunk_num = op->sector_num / sectors_per_chunk;

    nb_chunks = DIV_ROUND_UP(op->nb_sectors, sectors_per_chunk);

    bitmap_clear(s->in_flight_bitmap, chunk_num, nb_chunks);

    if (ret >= 0) {

        if (s->cow_bitmap) {

            bitmap_set(s->cow_bitmap, chunk_num, nb_chunks);

        }

        s->common.offset += (uint64_t)op->nb_sectors * BDRV_SECTOR_SIZE;

    }



    qemu_iovec_destroy(&op->qiov);

    g_free(op);



    if (s->waiting_for_io) {

        qemu_coroutine_enter(s->common.co, NULL);

    }

}
