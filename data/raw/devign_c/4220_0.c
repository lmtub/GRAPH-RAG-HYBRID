static void fd_accept_incoming_migration(void *opaque)

{

    QEMUFile *f = opaque;

    int ret;



    ret = qemu_loadvm_state(f);

    if (ret < 0) {

        fprintf(stderr, "load of migration failed\n");

        goto err;

    }

    qemu_announce_self();

    DPRINTF("successfully loaded vm state\n");

    /* we've successfully migrated, close the fd */

    qemu_set_fd_handler2(qemu_stdio_fd(f), NULL, NULL, NULL, NULL);

    if (autostart)

        vm_start();



err:

    qemu_fclose(f);

}
