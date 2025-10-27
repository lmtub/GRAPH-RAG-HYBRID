static int64_t migration_get_rate_limit(void *opaque)

{

    MigrationState *s = opaque;



    return s->xfer_limit;

}
