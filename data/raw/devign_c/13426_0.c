static int usb_host_handle_control(USBDevice *dev, USBPacket *p,

               int request, int value, int index, int length, uint8_t *data)

{

    USBHostDevice *s = DO_UPCAST(USBHostDevice, dev, dev);

    struct usbdevfs_urb *urb;

    AsyncURB *aurb;

    int ret;



    /*

     * Process certain standard device requests.

     * These are infrequent and are processed synchronously.

     */



    /* Note request is (bRequestType << 8) | bRequest */

    trace_usb_host_req_control(s->bus_num, s->addr, request, value, index);



    switch (request) {

    case DeviceOutRequest | USB_REQ_SET_ADDRESS:

        return usb_host_set_address(s, value);



    case DeviceOutRequest | USB_REQ_SET_CONFIGURATION:

        return usb_host_set_config(s, value & 0xff);



    case InterfaceOutRequest | USB_REQ_SET_INTERFACE:

        return usb_host_set_interface(s, index, value);

    }



    /* The rest are asynchronous */



    if (length > sizeof(dev->data_buf)) {

        fprintf(stderr, "husb: ctrl buffer too small (%d > %zu)\n",

                length, sizeof(dev->data_buf));

        return USB_RET_STALL;

    }



    aurb = async_alloc(s);

    aurb->packet = p;



    /*

     * Setup ctrl transfer.

     *

     * s->ctrl is laid out such that data buffer immediately follows

     * 'req' struct which is exactly what usbdevfs expects.

     */

    urb = &aurb->urb;



    urb->type     = USBDEVFS_URB_TYPE_CONTROL;

    urb->endpoint = p->devep;



    urb->buffer        = &dev->setup_buf;

    urb->buffer_length = length + 8;



    urb->usercontext = s;



    trace_usb_host_urb_submit(s->bus_num, s->addr, aurb,

                              urb->buffer_length, aurb->more);

    ret = ioctl(s->fd, USBDEVFS_SUBMITURB, urb);



    DPRINTF("husb: submit ctrl. len %u aurb %p\n", urb->buffer_length, aurb);



    if (ret < 0) {

        DPRINTF("husb: submit failed. errno %d\n", errno);

        async_free(aurb);



        switch(errno) {

        case ETIMEDOUT:

            return USB_RET_NAK;

        case EPIPE:

        default:

            return USB_RET_STALL;

        }

    }



    return USB_RET_ASYNC;

}
