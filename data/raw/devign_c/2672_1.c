void ahci_init(AHCIState *s, DeviceState *qdev, AddressSpace *as, int ports)

{

    qemu_irq *irqs;

    int i;



    s->as = as;

    s->ports = ports;

    s->dev = g_malloc0(sizeof(AHCIDevice) * ports);

    ahci_reg_init(s);

    /* XXX BAR size should be 1k, but that breaks, so bump it to 4k for now */

    memory_region_init_io(&s->mem, OBJECT(qdev), &ahci_mem_ops, s,

                          "ahci", AHCI_MEM_BAR_SIZE);

    memory_region_init_io(&s->idp, OBJECT(qdev), &ahci_idp_ops, s,

                          "ahci-idp", 32);



    irqs = qemu_allocate_irqs(ahci_irq_set, s, s->ports);



    for (i = 0; i < s->ports; i++) {

        AHCIDevice *ad = &s->dev[i];



        ide_bus_new(&ad->port, sizeof(ad->port), qdev, i, 1);

        ide_init2(&ad->port, irqs[i]);



        ad->hba = s;

        ad->port_no = i;

        ad->port.dma = &ad->dma;

        ad->port.dma->ops = &ahci_dma_ops;

    }

}
