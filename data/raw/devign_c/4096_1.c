static char *vnc_socket_local_addr(const char *format, int fd) {

    struct sockaddr_storage sa;

    socklen_t salen;



    salen = sizeof(sa);

    if (getsockname(fd, (struct sockaddr*)&sa, &salen) < 0)

        return NULL;



    return addr_to_string(format, &sa, salen);

}
