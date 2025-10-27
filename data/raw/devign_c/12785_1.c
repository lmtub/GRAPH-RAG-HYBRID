static int gdbserver_open(int port)

{

    struct sockaddr_in sockaddr;

    int fd, val, ret;



    fd = socket(PF_INET, SOCK_STREAM, 0);

    if (fd < 0) {

        perror("socket");

        return -1;

    }

#ifndef _WIN32

    fcntl(fd, F_SETFD, FD_CLOEXEC);

#endif



    /* allow fast reuse */

    val = 1;

    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (char *)&val, sizeof(val));



    sockaddr.sin_family = AF_INET;

    sockaddr.sin_port = htons(port);

    sockaddr.sin_addr.s_addr = 0;

    ret = bind(fd, (struct sockaddr *)&sockaddr, sizeof(sockaddr));

    if (ret < 0) {

        perror("bind");


        return -1;

    }

    ret = listen(fd, 0);

    if (ret < 0) {

        perror("listen");


        return -1;

    }

    return fd;

}