static coroutine_fn int qcow2_co_pwrite_zeroes(BlockDriverState *bs,

    int64_t offset, int count, BdrvRequestFlags flags)

{

    int ret;

    BDRVQcow2State *s = bs->opaque;



    uint32_t head = offset % s->cluster_size;

    uint32_t tail = (offset + count) % s->cluster_size;



    trace_qcow2_pwrite_zeroes_start_req(qemu_coroutine_self(), offset, count);

    if (offset + count == bs->total_sectors * BDRV_SECTOR_SIZE) {

        tail = 0;

    }



    if (head || tail) {

        int64_t cl_start = (offset - head) >> BDRV_SECTOR_BITS;

        uint64_t off;

        unsigned int nr;



        assert(head + count <= s->cluster_size);



        /* check whether remainder of cluster already reads as zero */

        if (!(is_zero_sectors(bs, cl_start,

                              DIV_ROUND_UP(head, BDRV_SECTOR_SIZE)) &&

              is_zero_sectors(bs, (offset + count) >> BDRV_SECTOR_BITS,

                              DIV_ROUND_UP(-tail & (s->cluster_size - 1),

                                           BDRV_SECTOR_SIZE)))) {

            return -ENOTSUP;

        }



        qemu_co_mutex_lock(&s->lock);

        /* We can have new write after previous check */

        offset = cl_start << BDRV_SECTOR_BITS;

        count = s->cluster_size;

        nr = s->cluster_size;

        ret = qcow2_get_cluster_offset(bs, offset, &nr, &off);

        if (ret != QCOW2_CLUSTER_UNALLOCATED &&

            ret != QCOW2_CLUSTER_ZERO_PLAIN &&

            ret != QCOW2_CLUSTER_ZERO_ALLOC) {

            qemu_co_mutex_unlock(&s->lock);

            return -ENOTSUP;

        }

    } else {

        qemu_co_mutex_lock(&s->lock);

    }



    trace_qcow2_pwrite_zeroes(qemu_coroutine_self(), offset, count);



    /* Whatever is left can use real zero clusters */

    ret = qcow2_zero_clusters(bs, offset, count >> BDRV_SECTOR_BITS, flags);

    qemu_co_mutex_unlock(&s->lock);



    return ret;

}
