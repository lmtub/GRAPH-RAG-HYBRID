static void enable_device(PIIX4PMState *s, int slot)

{

    s->ar.gpe.sts[0] |= PIIX4_PCI_HOTPLUG_STATUS;

    s->pci0_status.up |= (1 << slot);

}
