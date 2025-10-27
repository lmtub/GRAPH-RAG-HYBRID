static void cb_hmp_change_bdrv_pwd(Monitor *mon, const char *password,

                                   void *opaque)

{

    Error *encryption_err = opaque;

    Error *err = NULL;

    const char *device;



    device = error_get_field(encryption_err, "device");



    qmp_block_passwd(device, password, &err);

    hmp_handle_error(mon, &err);

    error_free(encryption_err);



    monitor_read_command(mon, 1);

}
