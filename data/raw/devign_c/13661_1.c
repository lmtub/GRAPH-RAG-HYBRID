connect_to_qemu(

    const char *host,

    const char *port

) {

    struct addrinfo hints;

    struct addrinfo *server;

    int ret, sock;



    sock = qemu_socket(AF_INET, SOCK_STREAM, 0);

    if (sock < 0) {

        /* Error */

        fprintf(stderr, "Error opening socket!\n");

        return -1;

    }



    memset(&hints, 0, sizeof(struct addrinfo));

    hints.ai_family = AF_UNSPEC;

    hints.ai_socktype = SOCK_STREAM;

    hints.ai_flags = 0;

    hints.ai_protocol = 0;          /* Any protocol */



    ret = getaddrinfo(host, port, &hints, &server);



    if (ret != 0) {

        /* Error */

        fprintf(stderr, "getaddrinfo failed\n");

        return -1;

    }



    if (connect(sock, server->ai_addr, server->ai_addrlen) < 0) {

        /* Error */

        fprintf(stderr, "Could not connect\n");

        return -1;

    }

    if (verbose) {

        printf("Connected (sizeof Header=%zd)!\n", sizeof(VSCMsgHeader));

    }

    return sock;

}
