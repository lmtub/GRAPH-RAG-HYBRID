static void qcow_close(BlockDriverState *bs)

{

    BDRVQcowState *s = bs->opaque;



    qcrypto_cipher_free(s->cipher);

    s->cipher = NULL;

    g_free(s->l1_table);

    qemu_vfree(s->l2_cache);

    g_free(s->cluster_cache);

    g_free(s->cluster_data);



    migrate_del_blocker(s->migration_blocker);

    error_free(s->migration_blocker);

}
