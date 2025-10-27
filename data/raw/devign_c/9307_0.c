void spapr_dt_events(void *fdt, uint32_t check_exception_irq)

{

    int event_sources, epow_events;

    uint32_t irq_ranges[] = {cpu_to_be32(check_exception_irq), cpu_to_be32(1)};

    uint32_t interrupts[] = {cpu_to_be32(check_exception_irq), 0};



    _FDT(event_sources = fdt_add_subnode(fdt, 0, "event-sources"));



    _FDT(fdt_setprop(fdt, event_sources, "interrupt-controller", NULL, 0));

    _FDT(fdt_setprop_cell(fdt, event_sources, "#interrupt-cells", 2));

    _FDT(fdt_setprop(fdt, event_sources, "interrupt-ranges",

                     irq_ranges, sizeof(irq_ranges)));



    _FDT(epow_events = fdt_add_subnode(fdt, event_sources, "epow-events"));

    _FDT(fdt_setprop(fdt, epow_events, "interrupts",

                     interrupts, sizeof(interrupts)));

}
