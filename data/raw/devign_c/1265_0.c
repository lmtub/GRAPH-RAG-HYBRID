static int ehci_init_transfer(EHCIPacket *p)

{

    uint32_t cpage, offset, bytes, plen;

    dma_addr_t page;

    USBBus *bus = &p->queue->ehci->bus;

    BusState *qbus = BUS(bus);



    cpage  = get_field(p->qtd.token, QTD_TOKEN_CPAGE);

    bytes  = get_field(p->qtd.token, QTD_TOKEN_TBYTES);

    offset = p->qtd.bufptr[0] & ~QTD_BUFPTR_MASK;

    qemu_sglist_init(&p->sgl, qbus->parent, 5, p->queue->ehci->as);



    while (bytes > 0) {

        if (cpage > 4) {

            fprintf(stderr, "cpage out of range (%d)\n", cpage);

            return -1;

        }



        page  = p->qtd.bufptr[cpage] & QTD_BUFPTR_MASK;

        page += offset;

        plen  = bytes;

        if (plen > 4096 - offset) {

            plen = 4096 - offset;

            offset = 0;

            cpage++;

        }



        qemu_sglist_add(&p->sgl, page, plen);

        bytes -= plen;

    }

    return 0;

}
