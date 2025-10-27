static int nbd_establish_connection(BlockDriverState *bs)

{

    BDRVNBDState *s = bs->opaque;

    int sock;

    int ret;

    off_t size;

    size_t blocksize;



    if (s->host_spec[0] == '/') {

        sock = unix_socket_outgoing(s->host_spec);

    } else {

        sock = tcp_socket_outgoing_spec(s->host_spec);

    }



    /* Failed to establish connection */

    if (sock < 0) {

        logout("Failed to establish connection to NBD server\n");

        return -errno;

    }



    /* NBD handshake */

    ret = nbd_receive_negotiate(sock, s->export_name, &s->nbdflags, &size,

                                &blocksize);

    if (ret < 0) {

        logout("Failed to negotiate with the NBD server\n");

        closesocket(sock);

        return ret;

    }



    /* Now that we're connected, set the socket to be non-blocking and

     * kick the reply mechanism.  */

    socket_set_nonblock(sock);

    qemu_aio_set_fd_handler(s->sock, nbd_reply_ready, NULL,

                            nbd_have_request, s);



    s->sock = sock;

    s->size = size;

    s->blocksize = blocksize;



    logout("Established connection with NBD server\n");

    return 0;

}
