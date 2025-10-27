static int pci_read_devaddr(Monitor *mon, const char *addr,

                            int *busp, unsigned *slotp)

{

    int dom;



    /* strip legacy tag */

    if (!strncmp(addr, "pci_addr=", 9)) {

        addr += 9;

    }

    if (pci_parse_devaddr(addr, &dom, busp, slotp, NULL)) {

        monitor_printf(mon, "Invalid pci address\n");

        return -1;

    }

    if (dom != 0) {

        monitor_printf(mon, "Multiple PCI domains not supported, use device_add\n");

        return -1;

    }

    return 0;

}
