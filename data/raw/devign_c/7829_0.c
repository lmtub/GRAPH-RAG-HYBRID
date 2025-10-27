int pci_device_load(PCIDevice *s, QEMUFile *f)

{

    uint8_t config[PCI_CONFIG_SPACE_SIZE];

    uint32_t version_id;

    int i;



    version_id = qemu_get_be32(f);

    if (version_id > 2)

        return -EINVAL;

    qemu_get_buffer(f, config, sizeof config);

    for (i = 0; i < sizeof config; ++i)

        if ((config[i] ^ s->config[i]) & s->cmask[i] & ~s->wmask[i])

            return -EINVAL;

    memcpy(s->config, config, sizeof config);



    pci_update_mappings(s);



    if (version_id >= 2)

        for (i = 0; i < 4; i ++)

            s->irq_state[i] = qemu_get_be32(f);

    return 0;

}
