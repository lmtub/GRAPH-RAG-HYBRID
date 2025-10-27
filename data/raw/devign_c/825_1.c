static void vmdk_free_extents(BlockDriverState *bs)

{

    int i;

    BDRVVmdkState *s = bs->opaque;



    for (i = 0; i < s->num_extents; i++) {

        g_free(s->extents[i].l1_table);

        g_free(s->extents[i].l2_cache);

        g_free(s->extents[i].l1_backup_table);

    }

    g_free(s->extents);

}
