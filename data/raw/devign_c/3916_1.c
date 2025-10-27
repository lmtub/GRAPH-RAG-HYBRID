MigrationParameters *qmp_query_migrate_parameters(Error **errp)

{

    MigrationParameters *params;

    MigrationState *s = migrate_get_current();



    params = g_malloc0(sizeof(*params));

    params->compress_level = s->parameters.compress_level;

    params->compress_threads = s->parameters.compress_threads;

    params->decompress_threads = s->parameters.decompress_threads;

    params->cpu_throttle_initial = s->parameters.cpu_throttle_initial;

    params->cpu_throttle_increment = s->parameters.cpu_throttle_increment;





    return params;

}