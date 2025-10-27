static int qcow2_make_empty(BlockDriverState *bs)

{

    BDRVQcow2State *s = bs->opaque;

    uint64_t offset, end_offset;

    int step = QEMU_ALIGN_DOWN(INT_MAX, s->cluster_size);

    int l1_clusters, ret = 0;



    l1_clusters = DIV_ROUND_UP(s->l1_size, s->cluster_size / sizeof(uint64_t));



    if (s->qcow_version >= 3 && !s->snapshots &&

        3 + l1_clusters <= s->refcount_block_size &&

        s->crypt_method_header != QCOW_CRYPT_LUKS) {

        /* The following function only works for qcow2 v3 images (it requires

         * the dirty flag) and only as long as there are no snapshots (because

         * it completely empties the image). Furthermore, the L1 table and three

         * additional clusters (image header, refcount table, one refcount

         * block) have to fit inside one refcount block. It cannot be used

         * for LUKS (yet) as it throws away the LUKS header cluster(s) */

        return make_completely_empty(bs);

    }



    /* This fallback code simply discards every active cluster; this is slow,

     * but works in all cases */

    end_offset = bs->total_sectors * BDRV_SECTOR_SIZE;

    for (offset = 0; offset < end_offset; offset += step) {

        /* As this function is generally used after committing an external

         * snapshot, QCOW2_DISCARD_SNAPSHOT seems appropriate. Also, the

         * default action for this kind of discard is to pass the discard,

         * which will ideally result in an actually smaller image file, as

         * is probably desired. */

        ret = qcow2_cluster_discard(bs, offset, MIN(step, end_offset - offset),

                                    QCOW2_DISCARD_SNAPSHOT, true);

        if (ret < 0) {

            break;

        }

    }



    return ret;

}
