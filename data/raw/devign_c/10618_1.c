static int get_pci_config_device(QEMUFile *f, void *pv, size_t size)

{

    PCIDevice *s = container_of(pv, PCIDevice, config);

    uint8_t config[size];

    int i;



    qemu_get_buffer(f, config, size);

    for (i = 0; i < size; ++i)

        if ((config[i] ^ s->config[i]) & s->cmask[i] & ~s->wmask[i])

            return -EINVAL;

    memcpy(s->config, config, size);



    pci_update_mappings(s);



    return 0;

}
