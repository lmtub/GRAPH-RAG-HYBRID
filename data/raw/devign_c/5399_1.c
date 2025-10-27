static void qcow2_close(BlockDriverState *bs)

{

    BDRVQcowState *s = bs->opaque;

    g_free(s->l1_table);

    /* else pre-write overlap checks in cache_destroy may crash */

    s->l1_table = NULL;



    if (!(bs->open_flags & BDRV_O_INCOMING)) {

        qcow2_cache_flush(bs, s->l2_table_cache);

        qcow2_cache_flush(bs, s->refcount_block_cache);



        qcow2_mark_clean(bs);

    }



    qcow2_cache_destroy(bs, s->l2_table_cache);

    qcow2_cache_destroy(bs, s->refcount_block_cache);



    g_free(s->unknown_header_fields);

    cleanup_unknown_header_ext(bs);



    g_free(s->cluster_cache);

    qemu_vfree(s->cluster_data);

    qcow2_refcount_close(bs);

    qcow2_free_snapshots(bs);

}
