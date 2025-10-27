static void unix_accept_incoming_migration(void *opaque)

{

    struct sockaddr_un addr;

    socklen_t addrlen = sizeof(addr);

    int s = (unsigned long)opaque;

    QEMUFile *f;

    int c, ret;



    do {

        c = qemu_accept(s, (struct sockaddr *)&addr, &addrlen);

    } while (c == -1 && socket_error() == EINTR);



    DPRINTF("accepted migration\n");



    if (c == -1) {

        fprintf(stderr, "could not accept migration connection\n");

        return;

    }



    f = qemu_fopen_socket(c);

    if (f == NULL) {

        fprintf(stderr, "could not qemu_fopen socket\n");

        goto out;

    }



    ret = qemu_loadvm_state(f);

    if (ret < 0) {

        fprintf(stderr, "load of migration failed\n");

        goto out_fopen;

    }

    qemu_announce_self();

    DPRINTF("successfully loaded vm state\n");



    /* we've successfully migrated, close the server socket */

    qemu_set_fd_handler2(s, NULL, NULL, NULL, NULL);

    close(s);



out_fopen:

    qemu_fclose(f);

out:

    close(c);

}
