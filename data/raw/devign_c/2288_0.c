int qcow2_alloc_clusters_at(BlockDriverState *bs, uint64_t offset,

    int nb_clusters)

{

    BDRVQcow2State *s = bs->opaque;

    uint64_t cluster_index, refcount;

    uint64_t i;

    int ret;



    assert(nb_clusters >= 0);

    if (nb_clusters == 0) {

        return 0;

    }



    do {

        /* Check how many clusters there are free */

        cluster_index = offset >> s->cluster_bits;

        for(i = 0; i < nb_clusters; i++) {

            ret = qcow2_get_refcount(bs, cluster_index++, &refcount);

            if (ret < 0) {

                return ret;

            } else if (refcount != 0) {

                break;

            }

        }



        /* And then allocate them */

        ret = update_refcount(bs, offset, i << s->cluster_bits, 1, false,

                              QCOW2_DISCARD_NEVER);

    } while (ret == -EAGAIN);



    if (ret < 0) {

        return ret;

    }



    return i;

}
