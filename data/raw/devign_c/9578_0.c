static int gdbserver_open(int port)

{

    struct sockaddr_in sockaddr;

    int fd, ret;



    fd = socket(PF_INET, SOCK_STREAM, 0);

    if (fd < 0) {

        perror("socket");

        return -1;

    }

#ifndef _WIN32

    fcntl(fd, F_SETFD, FD_CLOEXEC);

#endif



    socket_set_fast_reuse(fd);



    sockaddr.sin_family = AF_INET;

    sockaddr.sin_port = htons(port);

    sockaddr.sin_addr.s_addr = 0;

    ret = bind(fd, (struct sockaddr *)&sockaddr, sizeof(sockaddr));

    if (ret < 0) {

        perror("bind");

        close(fd);

        return -1;

    }

    ret = listen(fd, 0);

    if (ret < 0) {

        perror("listen");

        close(fd);

        return -1;

    }

    return fd;

}
