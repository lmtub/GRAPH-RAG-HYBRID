static int piix4_pm_initfn(PCIDevice *dev)

{

    PIIX4PMState *s = DO_UPCAST(PIIX4PMState, dev, dev);

    uint8_t *pci_conf;



    pci_conf = s->dev.config;

    pci_conf[0x06] = 0x80;

    pci_conf[0x07] = 0x02;

    pci_conf[0x09] = 0x00;

    pci_conf[0x3d] = 0x01; // interrupt pin 1



    /* APM */

    apm_init(dev, &s->apm, apm_ctrl_changed, s);



    if (s->kvm_enabled) {

        /* Mark SMM as already inited to prevent SMM from running.  KVM does not

         * support SMM mode. */

        pci_conf[0x5B] = 0x02;

    }



    /* XXX: which specification is used ? The i82731AB has different

       mappings */

    pci_conf[0x90] = s->smb_io_base | 1;

    pci_conf[0x91] = s->smb_io_base >> 8;

    pci_conf[0xd2] = 0x09;

    pm_smbus_init(&s->dev.qdev, &s->smb);

    memory_region_set_enabled(&s->smb.io, pci_conf[0xd2] & 1);

    memory_region_add_subregion(pci_address_space_io(dev),

                                s->smb_io_base, &s->smb.io);



    memory_region_init(&s->io, "piix4-pm", 64);

    memory_region_set_enabled(&s->io, false);

    memory_region_add_subregion(pci_address_space_io(dev),

                                0, &s->io);



    acpi_pm_tmr_init(&s->ar, pm_tmr_timer, &s->io);

    acpi_pm1_evt_init(&s->ar, pm_tmr_timer, &s->io);

    acpi_pm1_cnt_init(&s->ar, &s->io);

    acpi_gpe_init(&s->ar, GPE_LEN);



    s->powerdown_notifier.notify = piix4_pm_powerdown_req;

    qemu_register_powerdown_notifier(&s->powerdown_notifier);



    s->machine_ready.notify = piix4_pm_machine_ready;

    qemu_add_machine_init_done_notifier(&s->machine_ready);

    qemu_register_reset(piix4_reset, s);



    piix4_acpi_system_hot_add_init(pci_address_space_io(dev), dev->bus, s);



    return 0;

}
