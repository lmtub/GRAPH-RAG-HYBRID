void pci_device_save(PCIDevice *s, QEMUFile *f)

{

    int i;



    qemu_put_be32(f, s->version_id); /* PCI device version */

    qemu_put_buffer(f, s->config, 256);

    for (i = 0; i < 4; i++)

        qemu_put_be32(f, s->irq_state[i]);

}
