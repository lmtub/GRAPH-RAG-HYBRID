static coroutine_fn int sd_co_readv(BlockDriverState *bs, int64_t sector_num,

                       int nb_sectors, QEMUIOVector *qiov)

{

    SheepdogAIOCB acb;

    BDRVSheepdogState *s = bs->opaque;



    sd_aio_setup(&acb, s, qiov, sector_num, nb_sectors, AIOCB_READ_UDATA);



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
