void migrate_add_blocker(Error *reason)

{

    migration_blockers = g_slist_prepend(migration_blockers, reason);

}
