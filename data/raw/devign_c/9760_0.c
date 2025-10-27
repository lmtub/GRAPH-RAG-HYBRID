static void piix3_write_config(PCIDevice *dev,

                               uint32_t address, uint32_t val, int len)

{

    pci_default_write_config(dev, address, val, len);

    if (ranges_overlap(address, len, PIIX_PIRQC, 4)) {

        PIIX3State *piix3 = PIIX3_PCI_DEVICE(dev);

        int pic_irq;



        pci_bus_fire_intx_routing_notifier(piix3->dev.bus);

        piix3_update_irq_levels(piix3);

        for (pic_irq = 0; pic_irq < PIIX_NUM_PIC_IRQS; pic_irq++) {

            piix3_set_irq_pic(piix3, pic_irq);

        }

    }

}
