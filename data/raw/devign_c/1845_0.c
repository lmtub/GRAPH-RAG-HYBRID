void socket_listen_cleanup(int fd, Error **errp)

{

    SocketAddress *addr;



    addr = socket_local_address(fd, errp);



    if (addr->type == SOCKET_ADDRESS_KIND_UNIX

        && addr->u.q_unix.data->path) {

        if (unlink(addr->u.q_unix.data->path) < 0 && errno != ENOENT) {

            error_setg_errno(errp, errno,

                             "Failed to unlink socket %s",

                             addr->u.q_unix.data->path);

        }

    }



    qapi_free_SocketAddress(addr);

}
