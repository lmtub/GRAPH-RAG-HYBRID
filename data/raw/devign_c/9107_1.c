ssize_t migrate_fd_put_buffer(void *opaque, const void *data, size_t size)

{

    FdMigrationState *s = opaque;

    ssize_t ret;



    do {

        ret = s->write(s, data, size);

    } while (ret == -1 && ((s->get_error(s)) == EINTR));



    if (ret == -1)

        ret = -(s->get_error(s));



    if (ret == -EAGAIN) {

        qemu_set_fd_handler2(s->fd, NULL, NULL, migrate_fd_put_notify, s);

    } else if (ret < 0) {

        s->state = MIG_STATE_ERROR;

        notifier_list_notify(&migration_state_notifiers, NULL);

    }



    return ret;

}
