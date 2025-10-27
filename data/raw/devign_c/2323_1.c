static void pci_info_device(PCIBus *bus, PCIDevice *d)

{

    Monitor *mon = cur_mon;

    int i, class;

    PCIIORegion *r;

    const pci_class_desc *desc;



    monitor_printf(mon, "  Bus %2d, device %3d, function %d:\n",

                   pci_bus_num(d->bus),

                   PCI_SLOT(d->devfn), PCI_FUNC(d->devfn));

    class = pci_get_word(d->config + PCI_CLASS_DEVICE);

    monitor_printf(mon, "    ");

    desc = pci_class_descriptions;

    while (desc->desc && class != desc->class)

        desc++;

    if (desc->desc) {

        monitor_printf(mon, "%s", desc->desc);

    } else {

        monitor_printf(mon, "Class %04x", class);

    }

    monitor_printf(mon, ": PCI device %04x:%04x\n",

           pci_get_word(d->config + PCI_VENDOR_ID),

           pci_get_word(d->config + PCI_DEVICE_ID));



    if (d->config[PCI_INTERRUPT_PIN] != 0) {

        monitor_printf(mon, "      IRQ %d.\n",

                       d->config[PCI_INTERRUPT_LINE]);

    }

    if (class == 0x0604) {

        uint64_t base;

        uint64_t limit;



        monitor_printf(mon, "      BUS %d.\n", d->config[0x19]);

        monitor_printf(mon, "      secondary bus %d.\n",

                       d->config[PCI_SECONDARY_BUS]);

        monitor_printf(mon, "      subordinate bus %d.\n",

                       d->config[PCI_SUBORDINATE_BUS]);



        base = pci_bridge_get_base(d, PCI_BASE_ADDRESS_SPACE_IO);

        limit = pci_bridge_get_limit(d, PCI_BASE_ADDRESS_SPACE_IO);

        monitor_printf(mon, "      IO range [0x%04"PRIx64", 0x%04"PRIx64"]\n",

                       base, limit);



        base = pci_bridge_get_base(d, PCI_BASE_ADDRESS_SPACE_MEMORY);

        limit= pci_config_get_memory_base(d, PCI_BASE_ADDRESS_SPACE_MEMORY);

        monitor_printf(mon,

                       "      memory range [0x%08"PRIx64", 0x%08"PRIx64"]\n",

                       base, limit);



        base = pci_bridge_get_base(d, PCI_BASE_ADDRESS_SPACE_MEMORY |

                                   PCI_BASE_ADDRESS_MEM_PREFETCH);

        limit = pci_bridge_get_limit(d, PCI_BASE_ADDRESS_SPACE_MEMORY |

                                     PCI_BASE_ADDRESS_MEM_PREFETCH);

        monitor_printf(mon, "      prefetchable memory range "

                       "[0x%08"PRIx64", 0x%08"PRIx64"]\n", base, limit);

    }

    for(i = 0;i < PCI_NUM_REGIONS; i++) {

        r = &d->io_regions[i];

        if (r->size != 0) {

            monitor_printf(mon, "      BAR%d: ", i);

            if (r->type & PCI_BASE_ADDRESS_SPACE_IO) {

                monitor_printf(mon, "I/O at 0x%04"FMT_PCIBUS

                               " [0x%04"FMT_PCIBUS"].\n",

                               r->addr, r->addr + r->size - 1);

            } else {

                const char *type = r->type & PCI_BASE_ADDRESS_MEM_TYPE_64 ?

                    "64 bit" : "32 bit";

                const char *prefetch =

                    r->type & PCI_BASE_ADDRESS_MEM_PREFETCH ?

                    " prefetchable" : "";



                monitor_printf(mon, "%s%s memory at 0x%08"FMT_PCIBUS

                               " [0x%08"FMT_PCIBUS"].\n",

                               type, prefetch,

                               r->addr, r->addr + r->size - 1);

            }

        }

    }

    monitor_printf(mon, "      id \"%s\"\n", d->qdev.id ? d->qdev.id : "");

    if (class == 0x0604 && d->config[0x19] != 0) {

        pci_for_each_device(bus, d->config[0x19], pci_info_device);

    }

}
