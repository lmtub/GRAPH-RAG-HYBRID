void ahci_init(AHCIState *s, DeviceState *qdev, DMAContext *dma, int ports)

{

    qemu_irq *irqs;

    int i;



    s->dma = dma;

    s->ports = ports;

    s->dev = g_malloc0(sizeof(AHCIDevice) * ports);

    ahci_reg_init(s);

    /* XXX BAR size should be 1k, but that breaks, so bump it to 4k for now */

    memory_region_init_io(&s->mem, &ahci_mem_ops, s, "ahci", AHCI_MEM_BAR_SIZE);

    memory_region_init_io(&s->idp, &ahci_idp_ops, s, "ahci-idp", 32);



    irqs = qemu_allocate_irqs(ahci_irq_set, s, s->ports);



    for (i = 0; i < s->ports; i++) {

        AHCIDevice *ad = &s->dev[i];



        ide_bus_new(&ad->port, qdev, i);

        ide_init2(&ad->port, irqs[i]);



        ad->hba = s;

        ad->port_no = i;

        ad->port.dma = &ad->dma;

        ad->port.dma->ops = &ahci_dma_ops;

        ad->port_regs.cmd = PORT_CMD_SPIN_UP | PORT_CMD_POWER_ON;

    }

}
