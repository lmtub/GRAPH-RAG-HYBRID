int qcow2_zero_clusters(BlockDriverState *bs, uint64_t offset, int nb_sectors)

{

    BDRVQcow2State *s = bs->opaque;

    unsigned int nb_clusters;

    int ret;



    /* The zero flag is only supported by version 3 and newer */

    if (s->qcow_version < 3) {

        return -ENOTSUP;

    }



    /* Each L2 table is handled by its own loop iteration */

    nb_clusters = size_to_clusters(s, nb_sectors << BDRV_SECTOR_BITS);



    s->cache_discards = true;



    while (nb_clusters > 0) {

        ret = zero_single_l2(bs, offset, nb_clusters);

        if (ret < 0) {

            goto fail;

        }



        nb_clusters -= ret;

        offset += (ret * s->cluster_size);

    }



    ret = 0;

fail:

    s->cache_discards = false;

    qcow2_process_discards(bs, ret);



    return ret;

}
