static void vnc_listen_read(void *opaque)

{

    VncDisplay *vs = opaque;

    struct sockaddr_in addr;

    socklen_t addrlen = sizeof(addr);



    /* Catch-up */

    vga_hw_update();



    int csock = accept(vs->lsock, (struct sockaddr *)&addr, &addrlen);

    if (csock != -1) {

        vnc_connect(vs, csock);

    }

}
