static coroutine_fn int qcow2_co_write_zeroes(BlockDriverState *bs,

    int64_t sector_num, int nb_sectors, BdrvRequestFlags flags)

{

    int ret;

    BDRVQcow2State *s = bs->opaque;



    int head = sector_num % s->cluster_sectors;

    int tail = (sector_num + nb_sectors) % s->cluster_sectors;



    trace_qcow2_write_zeroes_start_req(qemu_coroutine_self(), sector_num,

                                       nb_sectors);



    if (head != 0 || tail != 0) {

        int64_t cl_start = sector_num - head;



        assert(cl_start + s->cluster_sectors >= sector_num + nb_sectors);



        sector_num = cl_start;

        nb_sectors = s->cluster_sectors;



        if (!is_zero_cluster(bs, sector_num)) {

            return -ENOTSUP;

        }



        qemu_co_mutex_lock(&s->lock);

        /* We can have new write after previous check */

        if (!is_zero_cluster_top_locked(bs, sector_num)) {

            qemu_co_mutex_unlock(&s->lock);

            return -ENOTSUP;

        }

    } else {

        qemu_co_mutex_lock(&s->lock);

    }



    trace_qcow2_write_zeroes(qemu_coroutine_self(), sector_num, nb_sectors);



    /* Whatever is left can use real zero clusters */

    ret = qcow2_zero_clusters(bs, sector_num << BDRV_SECTOR_BITS, nb_sectors);

    qemu_co_mutex_unlock(&s->lock);



    return ret;

}
