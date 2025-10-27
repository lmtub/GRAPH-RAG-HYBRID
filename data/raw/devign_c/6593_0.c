void pci_device_hot_add(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    PCIDevice *dev = NULL;

    const char *pci_addr = qdict_get_str(qdict, "pci_addr");

    const char *type = qdict_get_str(qdict, "type");

    const char *opts = qdict_get_try_str(qdict, "opts");



    /* strip legacy tag */

    if (!strncmp(pci_addr, "pci_addr=", 9)) {

        pci_addr += 9;

    }



    if (!opts) {

        opts = "";

    }



    if (!strcmp(pci_addr, "auto"))

        pci_addr = NULL;



    if (strcmp(type, "nic") == 0)

        dev = qemu_pci_hot_add_nic(mon, pci_addr, opts);

    else if (strcmp(type, "storage") == 0)

        dev = qemu_pci_hot_add_storage(mon, pci_addr, opts);

    else

        monitor_printf(mon, "invalid type: %s\n", type);



    if (dev) {

        *ret_data =

        qobject_from_jsonf("{ 'domain': 0, 'bus': %d, 'slot': %d, "

                           "'function': %d }", pci_bus_num(dev->bus),

                           PCI_SLOT(dev->devfn), PCI_FUNC(dev->devfn));

        assert(*ret_data != NULL);

    } else

        monitor_printf(mon, "failed to add %s\n", opts);

}
