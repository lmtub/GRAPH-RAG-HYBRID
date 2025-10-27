static BlockDriverAIOCB *bdrv_aio_rw_vector(BlockDriverState *bs,

                                            int64_t sector_num,

                                            QEMUIOVector *iov,

                                            int nb_sectors,

                                            BlockDriverCompletionFunc *cb,

                                            void *opaque,

                                            int is_write)



{

    VectorTranslationState *s = qemu_mallocz(sizeof(*s));

    BlockDriverAIOCB *aiocb = qemu_aio_get(bs, cb, opaque);



    s->this_aiocb = aiocb;

    s->iov = iov;

    s->bounce = qemu_memalign(512, nb_sectors * 512);

    s->is_write = is_write;

    if (is_write) {

        qemu_iovec_to_buffer(s->iov, s->bounce);

        s->aiocb = bdrv_aio_write(bs, sector_num, s->bounce, nb_sectors,

                                  bdrv_aio_rw_vector_cb, s);

    } else {

        s->aiocb = bdrv_aio_read(bs, sector_num, s->bounce, nb_sectors,

                                 bdrv_aio_rw_vector_cb, s);

    }

    return aiocb;

}
