static coroutine_fn int sd_co_writev(BlockDriverState *bs, int64_t sector_num,

                        int nb_sectors, QEMUIOVector *qiov)

{

    SheepdogAIOCB *acb;

    int ret;



    if (bs->growable && sector_num + nb_sectors > bs->total_sectors) {

        ret = sd_truncate(bs, (sector_num + nb_sectors) * BDRV_SECTOR_SIZE);

        if (ret < 0) {

            return ret;

        }

        bs->total_sectors = sector_num + nb_sectors;

    }



    acb = sd_aio_setup(bs, qiov, sector_num, nb_sectors);

    acb->aio_done_func = sd_write_done;

    acb->aiocb_type = AIOCB_WRITE_UDATA;



    ret = sd_co_rw_vector(acb);

    if (ret <= 0) {

        qemu_aio_release(acb);

        return ret;

    }



    qemu_coroutine_yield();



    return acb->ret;

}
