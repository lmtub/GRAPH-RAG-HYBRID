static uint32_t virtio_read_config(PCIDevice *pci_dev,

                                   uint32_t address, int len)

{

    VirtIOPCIProxy *proxy = DO_UPCAST(VirtIOPCIProxy, pci_dev, pci_dev);

    struct virtio_pci_cfg_cap *cfg;



    if (proxy->config_cap &&

        ranges_overlap(address, len, proxy->config_cap + offsetof(struct virtio_pci_cfg_cap,

                                                                  pci_cfg_data),

                       sizeof cfg->pci_cfg_data)) {

        uint32_t off;

        uint32_t len;



        cfg = (void *)(proxy->pci_dev.config + proxy->config_cap);

        off = le32_to_cpu(cfg->cap.offset);

        len = le32_to_cpu(cfg->cap.length);



        if (len <= sizeof cfg->pci_cfg_data) {

            virtio_address_space_read(&proxy->modern_as, off,

                                      cfg->pci_cfg_data, len);

        }

    }



    return pci_default_read_config(pci_dev, address, len);

}
