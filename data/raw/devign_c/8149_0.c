struct USBEndpoint *usb_ep_get(USBDevice *dev, int pid, int ep)

{

    struct USBEndpoint *eps = pid == USB_TOKEN_IN ? dev->ep_in : dev->ep_out;

    if (ep == 0) {

        return &dev->ep_ctl;

    }

    assert(pid == USB_TOKEN_IN || pid == USB_TOKEN_OUT);

    assert(ep > 0 && ep <= USB_MAX_ENDPOINTS);

    return eps + ep - 1;

}
