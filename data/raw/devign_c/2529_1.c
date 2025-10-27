static void vdi_close(BlockDriverState *bs)

{

    BDRVVdiState *s = bs->opaque;



    g_free(s->bmap);



    migrate_del_blocker(s->migration_blocker);

    error_free(s->migration_blocker);

}
