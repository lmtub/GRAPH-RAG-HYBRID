static int usb_host_handle_data(USBDevice *dev, USBPacket *p)

{

    USBHostDevice *s = (USBHostDevice *)dev;

    int ret, fd, mode;

    int one = 1, shortpacket = 0, timeout = 50;

    sigset_t new_mask, old_mask;

    uint8_t devep = p->devep;



    /* protect data transfers from SIGALRM signal */

    sigemptyset(&new_mask);

    sigaddset(&new_mask, SIGALRM);

    sigprocmask(SIG_BLOCK, &new_mask, &old_mask);



    if (p->pid == USB_TOKEN_IN) {

        devep |= 0x80;

        mode = O_RDONLY;

        shortpacket = 1;

    } else {

        mode = O_WRONLY;

    }



    fd = ensure_ep_open(s, devep, mode);

    if (fd < 0) {

        sigprocmask(SIG_SETMASK, &old_mask, NULL);

        return USB_RET_NODEV;

    }



    if (ioctl(fd, USB_SET_TIMEOUT, &timeout) < 0) {

#ifdef DEBUG

        printf("handle_data: failed to set timeout - %s\n",

               strerror(errno));

#endif

    }



    if (shortpacket) {

        if (ioctl(fd, USB_SET_SHORT_XFER, &one) < 0) {

#ifdef DEBUG

            printf("handle_data: failed to set short xfer mode - %s\n",

                   strerror(errno));

#endif

            sigprocmask(SIG_SETMASK, &old_mask, NULL);

        }

    }



    if (p->pid == USB_TOKEN_IN)

        ret = read(fd, p->data, p->len);

    else

        ret = write(fd, p->data, p->len);



    sigprocmask(SIG_SETMASK, &old_mask, NULL);



    if (ret < 0) {

#ifdef DEBUG

        printf("handle_data: error after %s data - %s\n",

               pid == USB_TOKEN_IN ? "reading" : "writing", strerror(errno));

#endif

        switch(errno) {

        case ETIMEDOUT:

        case EINTR:

            return USB_RET_NAK;

        default:

            return USB_RET_STALL;

        }

    } else {

        return ret;

    }

}
