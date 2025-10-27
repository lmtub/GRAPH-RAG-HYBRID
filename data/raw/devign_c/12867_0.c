static void do_boot_set(Monitor *mon, const QDict *qdict)

{

    int res;

    const char *bootdevice = qdict_get_str(qdict, "bootdevice");



    res = qemu_boot_set(bootdevice);

    if (res == 0) {

        monitor_printf(mon, "boot device list now set to %s\n", bootdevice);

    } else if (res > 0) {

        monitor_printf(mon, "setting boot device list failed\n");

    } else {

        monitor_printf(mon, "no function defined to set boot device list for "

                       "this architecture\n");

    }

}
