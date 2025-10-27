void ich9_pm_init(PCIDevice *lpc_pci, ICH9LPCPMRegs *pm,

                  qemu_irq sci_irq)

{

    memory_region_init(&pm->io, OBJECT(lpc_pci), "ich9-pm", ICH9_PMIO_SIZE);

    memory_region_set_enabled(&pm->io, false);

    memory_region_add_subregion(pci_address_space_io(lpc_pci),

                                0, &pm->io);



    acpi_pm_tmr_init(&pm->acpi_regs, ich9_pm_update_sci_fn, &pm->io);

    acpi_pm1_evt_init(&pm->acpi_regs, ich9_pm_update_sci_fn, &pm->io);

    acpi_pm1_cnt_init(&pm->acpi_regs, &pm->io, pm->s4_val);



    acpi_gpe_init(&pm->acpi_regs, ICH9_PMIO_GPE0_LEN);

    memory_region_init_io(&pm->io_gpe, OBJECT(lpc_pci), &ich9_gpe_ops, pm,

                          "acpi-gpe0", ICH9_PMIO_GPE0_LEN);

    memory_region_add_subregion(&pm->io, ICH9_PMIO_GPE0_STS, &pm->io_gpe);



    memory_region_init_io(&pm->io_smi, OBJECT(lpc_pci), &ich9_smi_ops, pm,

                          "acpi-smi", 8);

    memory_region_add_subregion(&pm->io, ICH9_PMIO_SMI_EN, &pm->io_smi);



    pm->irq = sci_irq;

    qemu_register_reset(pm_reset, pm);

    pm->powerdown_notifier.notify = pm_powerdown_req;

    qemu_register_powerdown_notifier(&pm->powerdown_notifier);



    acpi_cpu_hotplug_init(pci_address_space_io(lpc_pci), OBJECT(lpc_pci),

                          &pm->gpe_cpu, ICH9_CPU_HOTPLUG_IO_BASE);



    if (pm->acpi_memory_hotplug.is_enabled) {

        acpi_memory_hotplug_init(pci_address_space_io(lpc_pci), OBJECT(lpc_pci),

                                 &pm->acpi_memory_hotplug);

    }

}
