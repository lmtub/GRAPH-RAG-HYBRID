void isa_register_portio_list(ISADevice *dev, uint16_t start,

                              const MemoryRegionPortio *pio_start,

                              void *opaque, const char *name)

{

    PortioList piolist;



    /* START is how we should treat DEV, regardless of the actual

       contents of the portio array.  This is how the old code

       actually handled e.g. the FDC device.  */

    isa_init_ioport(dev, start);



    /* FIXME: the device should store created PortioList in its state.  Note

       that DEV can be NULL here and that single device can register several

       portio lists.  Current implementation is leaking memory allocated

       in portio_list_init.  The leak is not critical because it happens only

       at initialization time.  */

    portio_list_init(&piolist, OBJECT(dev), pio_start, opaque, name);

    portio_list_add(&piolist, isabus->address_space_io, start);

}
