static void qdev_print_devinfo(DeviceClass *dc)

{

    error_printf("name \"%s\"", object_class_get_name(OBJECT_CLASS(dc)));

    if (dc->bus_type) {

        error_printf(", bus %s", dc->bus_type);

    }

    if (qdev_class_has_alias(dc)) {

        error_printf(", alias \"%s\"", qdev_class_get_alias(dc));

    }

    if (dc->desc) {

        error_printf(", desc \"%s\"", dc->desc);

    }

    if (dc->no_user) {

        error_printf(", no-user");

    }

    error_printf("\n");

}
