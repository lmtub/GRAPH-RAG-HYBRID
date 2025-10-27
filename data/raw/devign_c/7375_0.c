static void hmp_handle_error(Monitor *mon, Error **errp)

{

    if (error_is_set(errp)) {

        monitor_printf(mon, "%s\n", error_get_pretty(*errp));

        error_free(*errp);

    }

}
