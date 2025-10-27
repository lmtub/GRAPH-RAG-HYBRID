static void xhci_child_detach(USBPort *uport, USBDevice *child)

{

    USBBus *bus = usb_bus_from_device(child);

    XHCIState *xhci = container_of(bus, XHCIState, bus);



    xhci_detach_slot(xhci, uport);

}
