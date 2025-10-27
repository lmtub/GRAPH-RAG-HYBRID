bool migrate_zero_blocks(void)

{

    MigrationState *s;



    s = migrate_get_current();



    return s->enabled_capabilities[MIGRATION_CAPABILITY_ZERO_BLOCKS];

}
