static int get_pci_irq_state(QEMUFile *f, void *pv, size_t size)

{

    PCIDevice *s = container_of(pv, PCIDevice, config);

    uint32_t irq_state[PCI_NUM_PINS];

    int i;

    for (i = 0; i < PCI_NUM_PINS; ++i) {

        irq_state[i] = qemu_get_be32(f);

        if (irq_state[i] != 0x1 && irq_state[i] != 0) {

            fprintf(stderr, "irq state %d: must be 0 or 1.\n",

                    irq_state[i]);

            return -EINVAL;

        }

    }



    for (i = 0; i < PCI_NUM_PINS; ++i) {

        pci_set_irq_state(s, i, irq_state[i]);

    }



    return 0;

}
