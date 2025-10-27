MigrationState *migrate_init(const MigrationParams *params)

{

    MigrationState *s = migrate_get_current();

    int64_t bandwidth_limit = s->bandwidth_limit;

    bool enabled_capabilities[MIGRATION_CAPABILITY_MAX];

    int64_t xbzrle_cache_size = s->xbzrle_cache_size;

    int compress_level = s->parameters[MIGRATION_PARAMETER_COMPRESS_LEVEL];

    int compress_thread_count =

            s->parameters[MIGRATION_PARAMETER_COMPRESS_THREADS];

    int decompress_thread_count =

            s->parameters[MIGRATION_PARAMETER_DECOMPRESS_THREADS];

    int x_cpu_throttle_initial =

            s->parameters[MIGRATION_PARAMETER_X_CPU_THROTTLE_INITIAL];

    int x_cpu_throttle_increment =

            s->parameters[MIGRATION_PARAMETER_X_CPU_THROTTLE_INCREMENT];



    memcpy(enabled_capabilities, s->enabled_capabilities,

           sizeof(enabled_capabilities));



    memset(s, 0, sizeof(*s));

    s->params = *params;

    memcpy(s->enabled_capabilities, enabled_capabilities,

           sizeof(enabled_capabilities));

    s->xbzrle_cache_size = xbzrle_cache_size;



    s->parameters[MIGRATION_PARAMETER_COMPRESS_LEVEL] = compress_level;

    s->parameters[MIGRATION_PARAMETER_COMPRESS_THREADS] =

               compress_thread_count;

    s->parameters[MIGRATION_PARAMETER_DECOMPRESS_THREADS] =

               decompress_thread_count;

    s->parameters[MIGRATION_PARAMETER_X_CPU_THROTTLE_INITIAL] =

                x_cpu_throttle_initial;

    s->parameters[MIGRATION_PARAMETER_X_CPU_THROTTLE_INCREMENT] =

                x_cpu_throttle_increment;

    s->bandwidth_limit = bandwidth_limit;

    migrate_set_state(s, MIGRATION_STATUS_NONE, MIGRATION_STATUS_SETUP);



    QSIMPLEQ_INIT(&s->src_page_requests);



    s->total_time = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);

    return s;

}
