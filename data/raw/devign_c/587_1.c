static void fd_accept_incoming_migration(void *opaque)

{

    QEMUFile *f = opaque;



    qemu_set_fd_handler2(qemu_get_fd(f), NULL, NULL, NULL, NULL);

    process_incoming_migration(f);

}
