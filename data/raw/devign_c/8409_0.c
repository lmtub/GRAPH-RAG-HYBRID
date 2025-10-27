void qmp_migrate_set_speed(int64_t value, Error **errp)

{

    MigrationState *s;



    if (value < 0) {

        value = 0;

    }



    s = migrate_get_current();

    s->bandwidth_limit = value;

    qemu_file_set_rate_limit(s->file, s->bandwidth_limit);

}
