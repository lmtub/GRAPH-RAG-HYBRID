SocketAddressLegacy *socket_local_address(int fd, Error **errp)

{

    struct sockaddr_storage ss;

    socklen_t sslen = sizeof(ss);



    if (getsockname(fd, (struct sockaddr *)&ss, &sslen) < 0) {

        error_setg_errno(errp, errno, "%s",

                         "Unable to query local socket address");

        return NULL;

    }



    return socket_sockaddr_to_address(&ss, sslen, errp);

}
