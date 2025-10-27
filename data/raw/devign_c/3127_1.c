void qmp_xen_save_devices_state(const char *filename, Error **errp)

{

    QEMUFile *f;

    QIOChannelFile *ioc;

    int saved_vm_running;

    int ret;



    saved_vm_running = runstate_is_running();

    vm_stop(RUN_STATE_SAVE_VM);

    global_state_store_running();



    ioc = qio_channel_file_new_path(filename, O_WRONLY | O_CREAT, 0660, errp);

    if (!ioc) {

        goto the_end;

    }

    qio_channel_set_name(QIO_CHANNEL(ioc), "migration-xen-save-state");

    f = qemu_fopen_channel_output(QIO_CHANNEL(ioc));

    ret = qemu_save_device_state(f);

    qemu_fclose(f);

    if (ret < 0) {

        error_setg(errp, QERR_IO_ERROR);

    }



 the_end:

    if (saved_vm_running) {

        vm_start();

    }

}
