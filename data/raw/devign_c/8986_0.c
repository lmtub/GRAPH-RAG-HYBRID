static void *nbd_client_thread(void *arg)

{

    int fd = *(int *)arg;

    off_t size;

    size_t blocksize;

    uint32_t nbdflags;

    int sock;

    int ret;

    pthread_t show_parts_thread;



    do {

        sock = unix_socket_outgoing(sockpath);

        if (sock == -1) {

            goto out;

        }

    } while (sock == -1);



    ret = nbd_receive_negotiate(sock, NULL, &nbdflags,

                                &size, &blocksize);

    if (ret == -1) {

        goto out;

    }



    ret = nbd_init(fd, sock, nbdflags, size, blocksize);

    if (ret == -1) {

        goto out;

    }



    /* update partition table */

    pthread_create(&show_parts_thread, NULL, show_parts, NULL);



    if (verbose) {

        fprintf(stderr, "NBD device %s is now connected to %s\n",

                device, srcpath);

    } else {

        /* Close stderr so that the qemu-nbd process exits.  */

        dup2(STDOUT_FILENO, STDERR_FILENO);

    }



    ret = nbd_client(fd);

    if (ret) {

        goto out;

    }

    close(fd);

    kill(getpid(), SIGTERM);

    return (void *) EXIT_SUCCESS;



out:

    kill(getpid(), SIGTERM);

    return (void *) EXIT_FAILURE;

}
