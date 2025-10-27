static int uhci_handle_td(UHCIState *s, uint32_t addr, UHCI_TD *td, uint32_t *int_mask)

{

    UHCIAsync *async;

    int len = 0, max_len;

    uint8_t pid;



    /* Is active ? */

    if (!(td->ctrl & TD_CTRL_ACTIVE))

        return 1;



    async = uhci_async_find_td(s, addr, td->token);

    if (async) {

        /* Already submitted */

        async->valid = 32;



        if (!async->done)

            return 1;



        uhci_async_unlink(s, async);

        goto done;

    }



    /* Allocate new packet */

    async = uhci_async_alloc(s);

    if (!async)

        return 1;



    async->valid = 10;

    async->td    = addr;

    async->token = td->token;



    max_len = ((td->token >> 21) + 1) & 0x7ff;

    pid = td->token & 0xff;



    async->packet.pid     = pid;

    async->packet.devaddr = (td->token >> 8) & 0x7f;

    async->packet.devep   = (td->token >> 15) & 0xf;

    async->packet.data    = async->buffer;

    async->packet.len     = max_len;

    async->packet.complete_cb     = uhci_async_complete;

    async->packet.complete_opaque = s;



    switch(pid) {

    case USB_TOKEN_OUT:

    case USB_TOKEN_SETUP:

        cpu_physical_memory_read(td->buffer, async->buffer, max_len);

        len = uhci_broadcast_packet(s, &async->packet);

        if (len >= 0)

            len = max_len;

        break;



    case USB_TOKEN_IN:

        len = uhci_broadcast_packet(s, &async->packet);

        break;



    default:

        /* invalid pid : frame interrupted */

        uhci_async_free(s, async);

        s->status |= UHCI_STS_HCPERR;

        uhci_update_irq(s);

        return -1;

    }

 

    if (len == USB_RET_ASYNC) {

        uhci_async_link(s, async);

        return 2;

    }



    async->packet.len = len;



done:

    len = uhci_complete_td(s, td, async, int_mask);

    uhci_async_free(s, async);

    return len;

}
