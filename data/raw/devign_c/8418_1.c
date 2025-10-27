static int ahci_dma_prepare_buf(IDEDMA *dma, int is_write)

{

    AHCIDevice *ad = DO_UPCAST(AHCIDevice, dma, dma);

    IDEState *s = &ad->port.ifs[0];



    ahci_populate_sglist(ad, &s->sg);

    s->io_buffer_size = s->sg.size;



    DPRINTF(ad->port_no, "len=%#x\n", s->io_buffer_size);

    return s->io_buffer_size != 0;

}
