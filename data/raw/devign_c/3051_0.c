static coroutine_fn int qcow2_co_pdiscard(BlockDriverState *bs,

                                          int64_t offset, int count)

{

    int ret;

    BDRVQcow2State *s = bs->opaque;



    if (!QEMU_IS_ALIGNED(offset | count, s->cluster_size)) {

        assert(count < s->cluster_size);

        /* Ignore partial clusters, except for the special case of the

         * complete partial cluster at the end of an unaligned file */

        if (!QEMU_IS_ALIGNED(offset, s->cluster_size) ||

            offset + count != bs->total_sectors * BDRV_SECTOR_SIZE) {

            return -ENOTSUP;

        }

    }



    qemu_co_mutex_lock(&s->lock);

    ret = qcow2_discard_clusters(bs, offset, count >> BDRV_SECTOR_BITS,

                                 QCOW2_DISCARD_REQUEST, false);

    qemu_co_mutex_unlock(&s->lock);

    return ret;

}
