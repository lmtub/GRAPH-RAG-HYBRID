static UHCIAsync *uhci_async_alloc(UHCIState *s)

{

    UHCIAsync *async = g_malloc(sizeof(UHCIAsync));



    memset(&async->packet, 0, sizeof(async->packet));

    async->uhci  = s;

    async->valid = 0;

    async->td    = 0;

    async->token = 0;

    async->done  = 0;

    async->isoc  = 0;

    usb_packet_init(&async->packet);

    qemu_sglist_init(&async->sgl, 1);



    return async;

}
