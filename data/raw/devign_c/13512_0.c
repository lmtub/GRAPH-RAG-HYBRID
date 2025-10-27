static void coroutine_fn wait_for_overlapping_requests(BlockDriverState *bs,

        int64_t offset, unsigned int bytes)

{

    BdrvTrackedRequest *req;

    int64_t cluster_offset;

    unsigned int cluster_bytes;

    bool retry;



    /* If we touch the same cluster it counts as an overlap.  This guarantees

     * that allocating writes will be serialized and not race with each other

     * for the same cluster.  For example, in copy-on-read it ensures that the

     * CoR read and write operations are atomic and guest writes cannot

     * interleave between them.

     */

    round_bytes_to_clusters(bs, offset, bytes, &cluster_offset, &cluster_bytes);



    do {

        retry = false;

        QLIST_FOREACH(req, &bs->tracked_requests, list) {

            if (tracked_request_overlaps(req, cluster_offset, cluster_bytes)) {

                /* Hitting this means there was a reentrant request, for

                 * example, a block driver issuing nested requests.  This must

                 * never happen since it means deadlock.

                 */

                assert(qemu_coroutine_self() != req->co);



                qemu_co_queue_wait(&req->wait_queue);

                retry = true;

                break;

            }

        }

    } while (retry);

}
