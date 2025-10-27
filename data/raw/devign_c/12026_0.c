static void fadt_setup(AcpiFadtDescriptorRev1 *fadt, AcpiPmInfo *pm)

{

    fadt->model = 1;

    fadt->reserved1 = 0;

    fadt->sci_int = cpu_to_le16(pm->sci_int);

    fadt->smi_cmd = cpu_to_le32(ACPI_PORT_SMI_CMD);

    fadt->acpi_enable = pm->acpi_enable_cmd;

    fadt->acpi_disable = pm->acpi_disable_cmd;

    /* EVT, CNT, TMR offset matches hw/acpi/core.c */

    fadt->pm1a_evt_blk = cpu_to_le32(pm->io_base);

    fadt->pm1a_cnt_blk = cpu_to_le32(pm->io_base + 0x04);

    fadt->pm_tmr_blk = cpu_to_le32(pm->io_base + 0x08);

    fadt->gpe0_blk = cpu_to_le32(pm->gpe0_blk);

    /* EVT, CNT, TMR length matches hw/acpi/core.c */

    fadt->pm1_evt_len = 4;

    fadt->pm1_cnt_len = 2;

    fadt->pm_tmr_len = 4;

    fadt->gpe0_blk_len = pm->gpe0_blk_len;

    fadt->plvl2_lat = cpu_to_le16(0xfff); /* C2 state not supported */

    fadt->plvl3_lat = cpu_to_le16(0xfff); /* C3 state not supported */

    fadt->flags = cpu_to_le32((1 << ACPI_FADT_F_WBINVD) |

                              (1 << ACPI_FADT_F_PROC_C1) |

                              (1 << ACPI_FADT_F_SLP_BUTTON) |

                              (1 << ACPI_FADT_F_RTC_S4));

    fadt->flags |= cpu_to_le32(1 << ACPI_FADT_F_USE_PLATFORM_CLOCK);

    /* APIC destination mode ("Flat Logical") has an upper limit of 8 CPUs

     * For more than 8 CPUs, "Clustered Logical" mode has to be used

     */

    if (max_cpus > 8) {

        fadt->flags |= cpu_to_le32(1 << ACPI_FADT_F_FORCE_APIC_CLUSTER_MODEL);

    }

    fadt->century = RTC_CENTURY;

}
