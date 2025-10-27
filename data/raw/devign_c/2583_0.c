USBDevice *usb_host_device_open(const char *devname)

{

    int fd = -1, ret;

    USBHostDevice *dev = NULL;

    struct usbdevfs_connectinfo ci;

    char buf[1024];

    int bus_num, addr;

    char product_name[PRODUCT_NAME_SZ];



    dev = qemu_mallocz(sizeof(USBHostDevice));

    if (!dev)

        goto fail;



#ifdef DEBUG_ISOCH

    printf("usb_host_device_open %s\n", devname);

#endif

    if (usb_host_find_device(&bus_num, &addr,

                             product_name, sizeof(product_name),

                             devname) < 0)

        return NULL;



    snprintf(buf, sizeof(buf), USBDEVFS_PATH "/%03d/%03d",

             bus_num, addr);

    fd = open(buf, O_RDWR | O_NONBLOCK);

    if (fd < 0) {

        perror(buf);

        return NULL;

    }



    /* read the device description */

    dev->descr_len = read(fd, dev->descr, sizeof(dev->descr));

    if (dev->descr_len <= 0) {

        perror("usb_host_device_open: reading device data failed");

        goto fail;

    }



#ifdef DEBUG

    {

        int x;

        printf("=== begin dumping device descriptor data ===\n");

        for (x = 0; x < dev->descr_len; x++)

            printf("%02x ", dev->descr[x]);

        printf("\n=== end dumping device descriptor data ===\n");

    }

#endif



    dev->fd = fd;

    dev->configuration = 1;



    /* XXX - do something about initial configuration */

    if (!usb_host_update_interfaces(dev, 1))

        goto fail;



    ret = ioctl(fd, USBDEVFS_CONNECTINFO, &ci);

    if (ret < 0) {

        perror("usb_host_device_open: USBDEVFS_CONNECTINFO");

        goto fail;

    }



#ifdef DEBUG

    printf("host USB device %d.%d grabbed\n", bus_num, addr);

#endif



    ret = usb_linux_update_endp_table(dev);

    if (ret)

        goto fail;



    if (ci.slow)

        dev->dev.speed = USB_SPEED_LOW;

    else

        dev->dev.speed = USB_SPEED_HIGH;

    dev->dev.handle_packet = usb_generic_handle_packet;



    dev->dev.handle_reset = usb_host_handle_reset;

    dev->dev.handle_control = usb_host_handle_control;

    dev->dev.handle_data = usb_host_handle_data;

    dev->dev.handle_destroy = usb_host_handle_destroy;



    if (product_name[0] == '\0')

        snprintf(dev->dev.devname, sizeof(dev->dev.devname),

                 "host:%s", devname);

    else

        pstrcpy(dev->dev.devname, sizeof(dev->dev.devname),

                product_name);



#ifdef USE_ASYNCIO

    /* set up the signal handlers */

    sigemptyset(&sigact.sa_mask);

    sigact.sa_sigaction = isoch_done;

    sigact.sa_flags = SA_SIGINFO;

    sigact.sa_restorer = 0;

    ret = sigaction(SIG_ISOCOMPLETE, &sigact, NULL);

    if (ret < 0) {

        perror("usb_host_device_open: sigaction failed");

        goto fail;

    }



    if (pipe(dev->pipe_fds) < 0) {

        perror("usb_host_device_open: pipe creation failed");

        goto fail;

    }

    fcntl(dev->pipe_fds[0], F_SETFL, O_NONBLOCK | O_ASYNC);

    fcntl(dev->pipe_fds[1], F_SETFL, O_NONBLOCK);

    qemu_set_fd_handler(dev->pipe_fds[0], urb_completion_pipe_read, NULL, dev);

#endif

    dev->urbs_ready = 0;

    return (USBDevice *)dev;

fail:

    if (dev)

        qemu_free(dev);

    close(fd);

    return NULL;

}
