MigrationState *unix_start_outgoing_migration(Monitor *mon,

                                              const char *path,

					      int64_t bandwidth_limit,

					      int detach,

					      int blk,

					      int inc)

{

    FdMigrationState *s;

    struct sockaddr_un addr;

    int ret;



    addr.sun_family = AF_UNIX;

    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);



    s = qemu_mallocz(sizeof(*s));



    s->get_error = unix_errno;

    s->write = unix_write;

    s->close = unix_close;

    s->mig_state.cancel = migrate_fd_cancel;

    s->mig_state.get_status = migrate_fd_get_status;

    s->mig_state.release = migrate_fd_release;



    s->mig_state.blk = blk;

    s->mig_state.shared = inc;



    s->state = MIG_STATE_ACTIVE;

    s->mon = NULL;

    s->bandwidth_limit = bandwidth_limit;

    s->fd = socket(PF_UNIX, SOCK_STREAM, 0);

    if (s->fd < 0) {

        dprintf("Unable to open socket");

        goto err_after_alloc;

    }



    socket_set_nonblock(s->fd);



    if (!detach) {

        migrate_fd_monitor_suspend(s, mon);

    }



    do {

        ret = connect(s->fd, (struct sockaddr *)&addr, sizeof(addr));

        if (ret == -1)

	    ret = -(s->get_error(s));



        if (ret == -EINPROGRESS || ret == -EWOULDBLOCK)

	    qemu_set_fd_handler2(s->fd, NULL, NULL, unix_wait_for_connect, s);

    } while (ret == -EINTR);



    if (ret < 0 && ret != -EINPROGRESS && ret != -EWOULDBLOCK) {

        dprintf("connect failed\n");

        goto err_after_open;

    } else if (ret >= 0)

        migrate_fd_connect(s);



    return &s->mig_state;



err_after_open:

    close(s->fd);



err_after_alloc:

    qemu_free(s);

    return NULL;

}
