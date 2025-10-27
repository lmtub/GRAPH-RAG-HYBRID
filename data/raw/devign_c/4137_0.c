static void hmp_change_read_arg(Monitor *mon, const char *password,

                                void *opaque)

{

    qmp_change_vnc_password(password, NULL);

    monitor_read_command(mon, 1);

}
