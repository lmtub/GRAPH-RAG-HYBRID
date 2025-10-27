static void pci_change_irq_level(PCIDevice *pci_dev, int irq_num, int change)

{

    PCIBus *bus;

    for (;;) {

        bus = pci_dev->bus;

        irq_num = bus->map_irq(pci_dev, irq_num);

        if (bus->set_irq)

            break;

        pci_dev = bus->parent_dev;

    }

    bus->irq_count[irq_num] += change;

    bus->set_irq(bus->irq_opaque, irq_num, bus->irq_count[irq_num] != 0);

}
