void spapr_events_fdt_skel(void *fdt, uint32_t epow_irq)

{

    uint32_t epow_irq_ranges[] = {cpu_to_be32(epow_irq), cpu_to_be32(1)};

    uint32_t epow_interrupts[] = {cpu_to_be32(epow_irq), 0};



    _FDT((fdt_begin_node(fdt, "event-sources")));



    _FDT((fdt_property(fdt, "interrupt-controller", NULL, 0)));

    _FDT((fdt_property_cell(fdt, "#interrupt-cells", 2)));

    _FDT((fdt_property(fdt, "interrupt-ranges",

                       epow_irq_ranges, sizeof(epow_irq_ranges))));



    _FDT((fdt_begin_node(fdt, "epow-events")));

    _FDT((fdt_property(fdt, "interrupts",

                       epow_interrupts, sizeof(epow_interrupts))));

    _FDT((fdt_end_node(fdt)));



    _FDT((fdt_end_node(fdt)));

}
