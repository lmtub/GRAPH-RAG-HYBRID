MigrationState *migrate_get_current(void)

{

    static MigrationState current_migration = {

        .state = MIG_STATE_NONE,

        .bandwidth_limit = MAX_THROTTLE,

        .xbzrle_cache_size = DEFAULT_MIGRATE_CACHE_SIZE,

        .mbps = -1,

    };



    return &current_migration;

}
