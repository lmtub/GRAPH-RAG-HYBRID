SocketAddress *socket_remote_address(int fd, Error **errp)

{

    struct sockaddr_storage ss;

    socklen_t sslen = sizeof(ss);



    if (getpeername(fd, (struct sockaddr *)&ss, &sslen) < 0) {

        error_setg_errno(errp, errno, "%s",

                         "Unable to query remote socket address");

        return NULL;

    }



    return socket_sockaddr_to_address(&ss, sslen, errp);

}
