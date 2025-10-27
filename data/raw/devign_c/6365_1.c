int qcow2_discard_clusters(BlockDriverState *bs, uint64_t offset,

    int nb_sectors, enum qcow2_discard_type type, bool full_discard)

{

    BDRVQcow2State *s = bs->opaque;

    uint64_t end_offset;

    uint64_t nb_clusters;

    int ret;



    end_offset = offset + (nb_sectors << BDRV_SECTOR_BITS);



    /* Round start up and end down */

    offset = align_offset(offset, s->cluster_size);

    end_offset = start_of_cluster(s, end_offset);



    if (offset > end_offset) {

        return 0;

    }



    nb_clusters = size_to_clusters(s, end_offset - offset);



    s->cache_discards = true;



    /* Each L2 table is handled by its own loop iteration */

    while (nb_clusters > 0) {

        ret = discard_single_l2(bs, offset, nb_clusters, type, full_discard);

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
