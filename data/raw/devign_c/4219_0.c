build_gtdt(GArray *table_data, GArray *linker)

{

    int gtdt_start = table_data->len;

    AcpiGenericTimerTable *gtdt;



    gtdt = acpi_data_push(table_data, sizeof *gtdt);

    /* The interrupt values are the same with the device tree when adding 16 */

    gtdt->secure_el1_interrupt = ARCH_TIMER_S_EL1_IRQ + 16;

    gtdt->secure_el1_flags = ACPI_EDGE_SENSITIVE;



    gtdt->non_secure_el1_interrupt = ARCH_TIMER_NS_EL1_IRQ + 16;

    gtdt->non_secure_el1_flags = ACPI_EDGE_SENSITIVE;



    gtdt->virtual_timer_interrupt = ARCH_TIMER_VIRT_IRQ + 16;

    gtdt->virtual_timer_flags = ACPI_EDGE_SENSITIVE;



    gtdt->non_secure_el2_interrupt = ARCH_TIMER_NS_EL2_IRQ + 16;

    gtdt->non_secure_el2_flags = ACPI_EDGE_SENSITIVE;



    build_header(linker, table_data,

                 (void *)(table_data->data + gtdt_start), "GTDT",

                 table_data->len - gtdt_start, 2, NULL);

}
