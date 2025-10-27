static coroutine_fn int sd_co_pdiscard(BlockDriverState *bs, int64_t offset,

                                      int count)

{

    SheepdogAIOCB acb;

    BDRVSheepdogState *s = bs->opaque;

    QEMUIOVector discard_iov;

    struct iovec iov;

    uint32_t zero = 0;



    if (!s->discard_supported) {

        return 0;

    }



    memset(&discard_iov, 0, sizeof(discard_iov));

    memset(&iov, 0, sizeof(iov));

    iov.iov_base = &zero;

    iov.iov_len = sizeof(zero);

    discard_iov.iov = &iov;

    discard_iov.niov = 1;

    if (!QEMU_IS_ALIGNED(offset | count, BDRV_SECTOR_SIZE)) {

        return -ENOTSUP;

    }

    sd_aio_setup(&acb, s, &discard_iov, offset >> BDRV_SECTOR_BITS,

                 count >> BDRV_SECTOR_BITS, AIOCB_DISCARD_OBJ);



retry:

    if (check_overlapping_aiocb(s, &acb)) {

        qemu_co_queue_wait(&s->overlapping_queue);

        goto retry;

    }



    sd_co_rw_vector(&acb);



    QLIST_REMOVE(&acb, aiocb_siblings);

    qemu_co_queue_restart_all(&s->overlapping_queue);

    return acb.ret;

}
