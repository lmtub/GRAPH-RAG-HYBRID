void hmp_migrate_set_parameter(Monitor *mon, const QDict *qdict)

{

    const char *param = qdict_get_str(qdict, "parameter");

    const char *valuestr = qdict_get_str(qdict, "value");

    int64_t valuebw = 0;

    long valueint = 0;

    Error *err = NULL;

    bool use_int_value = false;

    int i;



    for (i = 0; i < MIGRATION_PARAMETER__MAX; i++) {

        if (strcmp(param, MigrationParameter_lookup[i]) == 0) {

            MigrationParameters p = { 0 };

            switch (i) {

            case MIGRATION_PARAMETER_COMPRESS_LEVEL:

                p.has_compress_level = true;

                use_int_value = true;

                break;

            case MIGRATION_PARAMETER_COMPRESS_THREADS:

                p.has_compress_threads = true;

                use_int_value = true;

                break;

            case MIGRATION_PARAMETER_DECOMPRESS_THREADS:

                p.has_decompress_threads = true;

                use_int_value = true;

                break;

            case MIGRATION_PARAMETER_CPU_THROTTLE_INITIAL:

                p.has_cpu_throttle_initial = true;

                use_int_value = true;

                break;

            case MIGRATION_PARAMETER_CPU_THROTTLE_INCREMENT:

                p.has_cpu_throttle_increment = true;

                use_int_value = true;

                break;

            case MIGRATION_PARAMETER_TLS_CREDS:

                p.has_tls_creds = true;

                p.tls_creds = (char *) valuestr;

                break;

            case MIGRATION_PARAMETER_TLS_HOSTNAME:

                p.has_tls_hostname = true;

                p.tls_hostname = (char *) valuestr;

                break;

            case MIGRATION_PARAMETER_MAX_BANDWIDTH:

                p.has_max_bandwidth = true;

                valuebw = qemu_strtosz_MiB(valuestr, NULL);

                if (valuebw < 0 || (size_t)valuebw != valuebw) {

                    error_setg(&err, "Invalid size %s", valuestr);

                    goto cleanup;

                }

                p.max_bandwidth = valuebw;

                break;

            case MIGRATION_PARAMETER_DOWNTIME_LIMIT:

                p.has_downtime_limit = true;

                use_int_value = true;

                break;

            case MIGRATION_PARAMETER_X_CHECKPOINT_DELAY:

                p.has_x_checkpoint_delay = true;

                use_int_value = true;

                break;

            }



            if (use_int_value) {

                if (qemu_strtol(valuestr, NULL, 10, &valueint) < 0) {

                    error_setg(&err, "Unable to parse '%s' as an int",

                               valuestr);

                    goto cleanup;

                }

                /* Set all integers; only one has_FOO will be set, and

                 * the code ignores the remaining values */

                p.compress_level = valueint;

                p.compress_threads = valueint;

                p.decompress_threads = valueint;

                p.cpu_throttle_initial = valueint;

                p.cpu_throttle_increment = valueint;

                p.downtime_limit = valueint;

                p.x_checkpoint_delay = valueint;

            }



            qmp_migrate_set_parameters(&p, &err);

            break;

        }

    }



    if (i == MIGRATION_PARAMETER__MAX) {

        error_setg(&err, QERR_INVALID_PARAMETER, param);

    }



 cleanup:

    if (err) {

        error_report_err(err);

    }

}
