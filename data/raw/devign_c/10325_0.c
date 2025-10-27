void qmp_migrate_set_downtime(double value, Error **errp)

{

    value *= 1e9;

    value = MAX(0, MIN(UINT64_MAX, value));

    max_downtime = (uint64_t)value;

}
