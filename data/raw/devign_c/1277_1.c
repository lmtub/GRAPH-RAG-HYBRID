static void mptsas_scsi_init(PCIDevice *dev, Error **errp)

{

    DeviceState *d = DEVICE(dev);

    MPTSASState *s = MPT_SAS(dev);



    dev->config[PCI_LATENCY_TIMER] = 0;

    dev->config[PCI_INTERRUPT_PIN] = 0x01;



    memory_region_init_io(&s->mmio_io, OBJECT(s), &mptsas_mmio_ops, s,

                          "mptsas-mmio", 0x4000);

    memory_region_init_io(&s->port_io, OBJECT(s), &mptsas_port_ops, s,

                          "mptsas-io", 256);

    memory_region_init_io(&s->diag_io, OBJECT(s), &mptsas_diag_ops, s,

                          "mptsas-diag", 0x10000);



    if (s->msi != ON_OFF_AUTO_OFF &&

        msi_init(dev, 0, 1, true, false) >= 0) {

        /* TODO check for errors */

        s->msi_in_use = true;

    }



    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_SPACE_IO, &s->port_io);

    pci_register_bar(dev, 1, PCI_BASE_ADDRESS_SPACE_MEMORY |

                                 PCI_BASE_ADDRESS_MEM_TYPE_32, &s->mmio_io);

    pci_register_bar(dev, 2, PCI_BASE_ADDRESS_SPACE_MEMORY |

                                 PCI_BASE_ADDRESS_MEM_TYPE_32, &s->diag_io);



    if (!s->sas_addr) {

        s->sas_addr = ((NAA_LOCALLY_ASSIGNED_ID << 24) |

                       IEEE_COMPANY_LOCALLY_ASSIGNED) << 36;

        s->sas_addr |= (pci_bus_num(dev->bus) << 16);

        s->sas_addr |= (PCI_SLOT(dev->devfn) << 8);

        s->sas_addr |= PCI_FUNC(dev->devfn);

    }

    s->max_devices = MPTSAS_NUM_PORTS;



    s->request_bh = qemu_bh_new(mptsas_fetch_requests, s);



    QTAILQ_INIT(&s->pending);



    scsi_bus_new(&s->bus, sizeof(s->bus), &dev->qdev, &mptsas_scsi_info, NULL);

    if (!d->hotplugged) {

        scsi_bus_legacy_handle_cmdline(&s->bus, errp);

    }

}
