int qdev_device_help(QemuOpts *opts)

{

    Error *local_err = NULL;

    const char *driver;

    DevicePropertyInfoList *prop_list;

    DevicePropertyInfoList *prop;



    driver = qemu_opt_get(opts, "driver");

    if (driver && is_help_option(driver)) {

        qdev_print_devinfos(false);

        return 1;

    }



    if (!driver || !qemu_opt_has_help_opt(opts)) {

        return 0;

    }



    qdev_get_device_class(&driver, &local_err);

    if (local_err) {

        goto error;

    }



    prop_list = qmp_device_list_properties(driver, &local_err);

    if (local_err) {

        goto error;

    }



    for (prop = prop_list; prop; prop = prop->next) {

        error_printf("%s.%s=%s", driver,

                     prop->value->name,

                     prop->value->type);

        if (prop->value->has_description) {

            error_printf(" (%s)\n", prop->value->description);

        } else {

            error_printf("\n");

        }

    }



    qapi_free_DevicePropertyInfoList(prop_list);

    return 1;



error:

    error_printf("%s\n", error_get_pretty(local_err));

    error_free(local_err);

    return 1;

}
