static void vpc_close(BlockDriverState *bs)

{

    BDRVVPCState *s = bs->opaque;

    g_free(s->pagetable);

#ifdef CACHE

    g_free(s->pageentry_u8);

#endif



    migrate_del_blocker(s->migration_blocker);

    error_free(s->migration_blocker);

}
