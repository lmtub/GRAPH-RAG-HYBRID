void qcow2_free_clusters(BlockDriverState *bs,

                          int64_t offset, int64_t size)

{

    int ret;



    BLKDBG_EVENT(bs->file, BLKDBG_CLUSTER_FREE);

    ret = update_refcount(bs, offset, size, -1);

    if (ret < 0) {

        fprintf(stderr, "qcow2_free_clusters failed: %s\n", strerror(-ret));

        abort();

    }

}
