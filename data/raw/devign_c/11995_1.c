int qcow2_alloc_cluster_offset(BlockDriverState *bs, uint64_t offset,

    int n_start, int n_end, int *num, uint64_t *host_offset, QCowL2Meta **m)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t start, remaining;

    uint64_t cluster_offset;

    uint64_t cur_bytes;

    int ret;



    trace_qcow2_alloc_clusters_offset(qemu_coroutine_self(), offset,

                                      n_start, n_end);



    assert(n_start * BDRV_SECTOR_SIZE == offset_into_cluster(s, offset));

    offset = start_of_cluster(s, offset);



again:

    start = offset + (n_start << BDRV_SECTOR_BITS);

    remaining = (n_end - n_start) << BDRV_SECTOR_BITS;

    cluster_offset = 0;

    *host_offset = 0;



    while (true) {

        /*

         * Now start gathering as many contiguous clusters as possible:

         *

         * 1. Check for overlaps with in-flight allocations

         *

         *      a) Overlap not in the first cluster -> shorten this request and

         *         let the caller handle the rest in its next loop iteration.

         *

         *      b) Real overlaps of two requests. Yield and restart the search

         *         for contiguous clusters (the situation could have changed

         *         while we were sleeping)

         *

         *      c) TODO: Request starts in the same cluster as the in-flight

         *         allocation ends. Shorten the COW of the in-fight allocation,

         *         set cluster_offset to write to the same cluster and set up

         *         the right synchronisation between the in-flight request and

         *         the new one.

         */

        cur_bytes = remaining;

        ret = handle_dependencies(bs, start, &cur_bytes);

        if (ret == -EAGAIN) {

            goto again;

        } else if (ret < 0) {

            return ret;

        } else {

            /* handle_dependencies() may have decreased cur_bytes (shortened

             * the allocations below) so that the next dependency is processed

             * correctly during the next loop iteration. */

        }



        /*

         * 2. Count contiguous COPIED clusters.

         */

        ret = handle_copied(bs, start, &cluster_offset, &cur_bytes, m);

        if (ret < 0) {

            return ret;

        } else if (ret) {

            if (!*host_offset) {

                *host_offset = start_of_cluster(s, cluster_offset);

            }



            start           += cur_bytes;

            remaining       -= cur_bytes;

            cluster_offset  += cur_bytes;



            cur_bytes = remaining;

        } else if (cur_bytes == 0) {

            break;

        }



        /* If there is something left to allocate, do that now */

        if (remaining == 0) {

            break;

        }



        /*

         * 3. If the request still hasn't completed, allocate new clusters,

         *    considering any cluster_offset of steps 1c or 2.

         */

        ret = handle_alloc(bs, start, &cluster_offset, &cur_bytes, m);

        if (ret < 0) {

            return ret;

        } else if (ret) {

            if (!*host_offset) {

                *host_offset = start_of_cluster(s, cluster_offset);

            }



            start           += cur_bytes;

            remaining       -= cur_bytes;

            cluster_offset  += cur_bytes;



            break;

        } else {

            assert(cur_bytes == 0);

            break;

        }

    }



    *num = (n_end - n_start) - (remaining >> BDRV_SECTOR_BITS);

    assert(*num > 0);

    assert(*host_offset != 0);



    return 0;

}
