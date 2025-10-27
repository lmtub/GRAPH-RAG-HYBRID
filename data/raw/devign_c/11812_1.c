void migrate_del_blocker(Error *reason)

{

    migration_blockers = g_slist_remove(migration_blockers, reason);

}
