static int usbredir_handle_data(USBDevice *udev, USBPacket *p)

{

    USBRedirDevice *dev = DO_UPCAST(USBRedirDevice, dev, udev);

    uint8_t ep;



    ep = p->devep;

    if (p->pid == USB_TOKEN_IN) {

        ep |= USB_DIR_IN;

    }



    switch (dev->endpoint[EP2I(ep)].type) {

    case USB_ENDPOINT_XFER_CONTROL:

        ERROR("handle_data called for control transfer on ep %02X\n", ep);

        return USB_RET_NAK;

    case USB_ENDPOINT_XFER_ISOC:

        return usbredir_handle_iso_data(dev, p, ep);

    case USB_ENDPOINT_XFER_BULK:

        return usbredir_handle_bulk_data(dev, p, ep);

    case USB_ENDPOINT_XFER_INT:

        return usbredir_handle_interrupt_data(dev, p, ep);

    default:

        ERROR("handle_data ep %02X has unknown type %d\n", ep,

              dev->endpoint[EP2I(ep)].type);

        return USB_RET_NAK;

    }

}
