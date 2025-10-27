static void bmdma_map(PCIDevice *pci_dev, int region_num,

                    pcibus_t addr, pcibus_t size, int type)

{

    PCIIDEState *d = DO_UPCAST(PCIIDEState, dev, pci_dev);

    int i;



    for(i = 0;i < 2; i++) {

        BMDMAState *bm = &d->bmdma[i];

        d->bus[i].bmdma = bm;

        bm->bus = d->bus+i;

        bm->pci_dev = d;

        qemu_add_vm_change_state_handler(ide_dma_restart_cb, bm);



        register_ioport_write(addr, 1, 1, bmdma_cmd_writeb, bm);



        register_ioport_write(addr + 1, 3, 1, bmdma_writeb, bm);

        register_ioport_read(addr, 4, 1, bmdma_readb, bm);



        register_ioport_write(addr + 4, 4, 1, bmdma_addr_writeb, bm);

        register_ioport_read(addr + 4, 4, 1, bmdma_addr_readb, bm);

        register_ioport_write(addr + 4, 4, 2, bmdma_addr_writew, bm);

        register_ioport_read(addr + 4, 4, 2, bmdma_addr_readw, bm);

        register_ioport_write(addr + 4, 4, 4, bmdma_addr_writel, bm);

        register_ioport_read(addr + 4, 4, 4, bmdma_addr_readl, bm);

        addr += 8;

    }

}
