static int connect_namedsocket(const char *path)

{

    int sockfd, size;

    struct sockaddr_un helper;



    if (strlen(path) >= sizeof(helper.sun_path)) {

        fprintf(stderr, "Socket name too large\n");

        return -1;

    }

    sockfd = socket(AF_UNIX, SOCK_STREAM, 0);

    if (sockfd < 0) {

        fprintf(stderr, "failed to create socket: %s\n", strerror(errno));

        return -1;

    }

    strcpy(helper.sun_path, path);

    helper.sun_family = AF_UNIX;

    size = strlen(helper.sun_path) + sizeof(helper.sun_family);

    if (connect(sockfd, (struct sockaddr *)&helper, size) < 0) {

        fprintf(stderr, "failed to connect to %s: %s\n", path, strerror(errno));

        close(sockfd);

        return -1;

    }



    /* remove the socket for security reasons */

    unlink(path);

    return sockfd;

}
