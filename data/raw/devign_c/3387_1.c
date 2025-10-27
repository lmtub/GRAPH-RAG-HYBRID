void hmp_change(Monitor *mon, const QDict *qdict)

{

    const char *device = qdict_get_str(qdict, "device");

    const char *target = qdict_get_str(qdict, "target");

    const char *arg = qdict_get_try_str(qdict, "arg");

    const char *read_only = qdict_get_try_str(qdict, "read-only-mode");

    BlockdevChangeReadOnlyMode read_only_mode = 0;

    Error *err = NULL;



    if (strcmp(device, "vnc") == 0) {

        if (read_only) {

            monitor_printf(mon,

                           "Parameter 'read-only-mode' is invalid for VNC\n");

            return;

        }

        if (strcmp(target, "passwd") == 0 ||

            strcmp(target, "password") == 0) {

            if (!arg) {

                monitor_read_password(mon, hmp_change_read_arg, NULL);

                return;

            }

        }

        qmp_change("vnc", target, !!arg, arg, &err);

    } else {

        if (read_only) {

            read_only_mode =

                qapi_enum_parse(BlockdevChangeReadOnlyMode_lookup,

                                read_only, BLOCKDEV_CHANGE_READ_ONLY_MODE__MAX,

                                BLOCKDEV_CHANGE_READ_ONLY_MODE_RETAIN, &err);

            if (err) {

                hmp_handle_error(mon, &err);

                return;

            }

        }



        qmp_blockdev_change_medium(true, device, false, NULL, target,

                                   !!arg, arg, !!read_only, read_only_mode,

                                   &err);

        if (err &&

            error_get_class(err) == ERROR_CLASS_DEVICE_ENCRYPTED) {

            error_free(err);

            monitor_read_block_device_key(mon, device, NULL, NULL);

            return;

        }

    }



    hmp_handle_error(mon, &err);

}
