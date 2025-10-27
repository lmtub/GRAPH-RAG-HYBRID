void qmp_nbd_server_stop(Error **errp)

{

    while (!QTAILQ_EMPTY(&close_notifiers)) {

        NBDCloseNotifier *cn = QTAILQ_FIRST(&close_notifiers);

        nbd_close_notifier(&cn->n, nbd_export_get_blockdev(cn->exp));

    }



    qemu_set_fd_handler2(server_fd, NULL, NULL, NULL, NULL);

    close(server_fd);

    server_fd = -1;

}
