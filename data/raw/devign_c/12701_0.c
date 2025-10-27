static int usb_host_init(void)

{

    const struct libusb_pollfd **poll;

    int i, rc;



    if (ctx) {

        return 0;

    }

    rc = libusb_init(&ctx);

    if (rc != 0) {

        return -1;

    }

    libusb_set_debug(ctx, loglevel);



    libusb_set_pollfd_notifiers(ctx, usb_host_add_fd,

                                usb_host_del_fd,

                                ctx);

    poll = libusb_get_pollfds(ctx);

    if (poll) {

        for (i = 0; poll[i] != NULL; i++) {

            usb_host_add_fd(poll[i]->fd, poll[i]->events, ctx);

        }

    }

    free(poll);

    return 0;

}
