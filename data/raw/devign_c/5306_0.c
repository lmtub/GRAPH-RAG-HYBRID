static int curl_sock_cb(CURL *curl, curl_socket_t fd, int action,

                        void *s, void *sp)

{

    DPRINTF("CURL (AIO): Sock action %d on fd %d\n", action, fd);

    switch (action) {

        case CURL_POLL_IN:

            qemu_aio_set_fd_handler(fd, curl_multi_do, NULL, s);

            break;

        case CURL_POLL_OUT:

            qemu_aio_set_fd_handler(fd, NULL, curl_multi_do, s);

            break;

        case CURL_POLL_INOUT:

            qemu_aio_set_fd_handler(fd, curl_multi_do, curl_multi_do, s);

            break;

        case CURL_POLL_REMOVE:

            qemu_aio_set_fd_handler(fd, NULL, NULL, NULL);

            break;

    }



    return 0;

}
