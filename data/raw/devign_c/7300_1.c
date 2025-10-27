static void megasas_scsi_realize(PCIDevice *dev, Error **errp)

{

    DeviceState *d = DEVICE(dev);

    MegasasState *s = MEGASAS(dev);

    MegasasBaseClass *b = MEGASAS_DEVICE_GET_CLASS(s);

    uint8_t *pci_conf;

    int i, bar_type;

    Error *err = NULL;

    int ret;



    pci_conf = dev->config;



    /* PCI latency timer = 0 */

    pci_conf[PCI_LATENCY_TIMER] = 0;

    /* Interrupt pin 1 */

    pci_conf[PCI_INTERRUPT_PIN] = 0x01;



    if (s->msi != ON_OFF_AUTO_OFF) {

        ret = msi_init(dev, 0x50, 1, true, false, &err);

        /* Any error other than -ENOTSUP(board's MSI support is broken)

         * is a programming error */

        assert(!ret || ret == -ENOTSUP);

        if (ret && s->msi == ON_OFF_AUTO_ON) {

            /* Can't satisfy user's explicit msi=on request, fail */

            error_append_hint(&err, "You have to use msi=auto (default) or "

                    "msi=off with this machine type.\n");

            error_propagate(errp, err);

            return;

        } else if (ret) {

            /* With msi=auto, we fall back to MSI off silently */

            s->msi = ON_OFF_AUTO_OFF;

            error_free(err);

        }

    }



    memory_region_init_io(&s->mmio_io, OBJECT(s), &megasas_mmio_ops, s,

                          "megasas-mmio", 0x4000);

    memory_region_init_io(&s->port_io, OBJECT(s), &megasas_port_ops, s,

                          "megasas-io", 256);

    memory_region_init_io(&s->queue_io, OBJECT(s), &megasas_queue_ops, s,

                          "megasas-queue", 0x40000);



    if (megasas_use_msix(s) &&

        msix_init(dev, 15, &s->mmio_io, b->mmio_bar, 0x2000,

                  &s->mmio_io, b->mmio_bar, 0x3800, 0x68)) {

        s->msix = ON_OFF_AUTO_OFF;

    }

    if (pci_is_express(dev)) {

        pcie_endpoint_cap_init(dev, 0xa0);

    }



    bar_type = PCI_BASE_ADDRESS_SPACE_MEMORY | PCI_BASE_ADDRESS_MEM_TYPE_64;

    pci_register_bar(dev, b->ioport_bar,

                     PCI_BASE_ADDRESS_SPACE_IO, &s->port_io);

    pci_register_bar(dev, b->mmio_bar, bar_type, &s->mmio_io);

    pci_register_bar(dev, 3, bar_type, &s->queue_io);



    if (megasas_use_msix(s)) {

        msix_vector_use(dev, 0);

    }



    s->fw_state = MFI_FWSTATE_READY;

    if (!s->sas_addr) {

        s->sas_addr = ((NAA_LOCALLY_ASSIGNED_ID << 24) |

                       IEEE_COMPANY_LOCALLY_ASSIGNED) << 36;

        s->sas_addr |= (pci_bus_num(dev->bus) << 16);

        s->sas_addr |= (PCI_SLOT(dev->devfn) << 8);

        s->sas_addr |= PCI_FUNC(dev->devfn);

    }

    if (!s->hba_serial) {

        s->hba_serial = g_strdup(MEGASAS_HBA_SERIAL);

    }

    if (s->fw_sge >= MEGASAS_MAX_SGE - MFI_PASS_FRAME_SIZE) {

        s->fw_sge = MEGASAS_MAX_SGE - MFI_PASS_FRAME_SIZE;

    } else if (s->fw_sge >= 128 - MFI_PASS_FRAME_SIZE) {

        s->fw_sge = 128 - MFI_PASS_FRAME_SIZE;

    } else {

        s->fw_sge = 64 - MFI_PASS_FRAME_SIZE;

    }

    if (s->fw_cmds > MEGASAS_MAX_FRAMES) {

        s->fw_cmds = MEGASAS_MAX_FRAMES;

    }

    trace_megasas_init(s->fw_sge, s->fw_cmds,

                       megasas_is_jbod(s) ? "jbod" : "raid");



    if (megasas_is_jbod(s)) {

        s->fw_luns = MFI_MAX_SYS_PDS;

    } else {

        s->fw_luns = MFI_MAX_LD;

    }

    s->producer_pa = 0;

    s->consumer_pa = 0;

    for (i = 0; i < s->fw_cmds; i++) {

        s->frames[i].index = i;

        s->frames[i].context = -1;

        s->frames[i].pa = 0;

        s->frames[i].state = s;

    }



    scsi_bus_new(&s->bus, sizeof(s->bus), DEVICE(dev),

                 &megasas_scsi_info, NULL);

    if (!d->hotplugged) {

        scsi_bus_legacy_handle_cmdline(&s->bus, errp);

    }

}
