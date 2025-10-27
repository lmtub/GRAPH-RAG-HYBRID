static int sh_pci_host_init(PCIDevice *d)

{

    pci_config_set_vendor_id(d->config, PCI_VENDOR_ID_HITACHI);

    pci_config_set_device_id(d->config, PCI_DEVICE_ID_HITACHI_SH7751R);

    pci_set_word(d->config + PCI_COMMAND, PCI_COMMAND_WAIT);

    pci_set_word(d->config + PCI_STATUS, PCI_STATUS_CAP_LIST |

                 PCI_STATUS_FAST_BACK | PCI_STATUS_DEVSEL_MEDIUM);

    return 0;

}
