static coroutine_fn int sd_co_discard(BlockDriverState *bs, int64_t sector_num,

                                      int nb_sectors)

{

    SheepdogAIOCB *acb;

    QEMUIOVector dummy;

    BDRVSheepdogState *s = bs->opaque;

    int ret;



    if (!s->discard_supported) {

            return 0;

    }



    acb = sd_aio_setup(bs, &dummy, sector_num, nb_sectors);

    acb->aiocb_type = AIOCB_DISCARD_OBJ;

    acb->aio_done_func = sd_finish_aiocb;



retry:

    if (check_overwrapping_aiocb(s, acb)) {

        qemu_co_queue_wait(&s->overwrapping_queue);

        goto retry;

    }



    ret = sd_co_rw_vector(acb);

    if (ret <= 0) {

        QLIST_REMOVE(acb, aiocb_siblings);

        qemu_co_queue_restart_all(&s->overwrapping_queue);

        qemu_aio_unref(acb);

        return ret;

    }



    qemu_coroutine_yield();



    QLIST_REMOVE(acb, aiocb_siblings);

    qemu_co_queue_restart_all(&s->overwrapping_queue);



    return acb->ret;

}
