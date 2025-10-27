void do_migrate_set_speed(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    double d;

    char *ptr;

    FdMigrationState *s;

    const char *value = qdict_get_str(qdict, "value");



    d = strtod(value, &ptr);

    switch (*ptr) {

    case 'G': case 'g':

        d *= 1024;

    case 'M': case 'm':

        d *= 1024;

    case 'K': case 'k':

        d *= 1024;

    default:

        break;

    }



    max_throttle = (uint32_t)d;

    s = migrate_to_fms(current_migration);



    if (s) {

        qemu_file_set_rate_limit(s->file, max_throttle);

    }

    

}
