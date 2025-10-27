static void disable_interrupt(EEPRO100State * s)

{

    if (s->int_stat) {

        logout("interrupt disabled\n");

        qemu_irq_lower(s->pci_dev->irq[0]);

        s->int_stat = 0;

    }

}
