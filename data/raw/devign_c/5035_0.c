static int64_t migration_set_rate_limit(void *opaque, int64_t new_rate)

{

    MigrationState *s = opaque;

    if (qemu_file_get_error(s->file)) {

        goto out;

    }



    s->xfer_limit = new_rate;



out:

    return s->xfer_limit;

}
