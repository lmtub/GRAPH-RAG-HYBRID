int spapr_populate_pci_dt(sPAPRPHBState *phb,

                          uint32_t xics_phandle,

                          void *fdt)

{

    int bus_off, i, j;

    char nodename[256];

    uint32_t bus_range[] = { cpu_to_be32(0), cpu_to_be32(0xff) };

    struct {

        uint32_t hi;

        uint64_t child;

        uint64_t parent;

        uint64_t size;

    } QEMU_PACKED ranges[] = {

        {

            cpu_to_be32(b_ss(1)), cpu_to_be64(0),

            cpu_to_be64(phb->io_win_addr),

            cpu_to_be64(memory_region_size(&phb->iospace)),

        },

        {

            cpu_to_be32(b_ss(2)), cpu_to_be64(SPAPR_PCI_MEM_WIN_BUS_OFFSET),

            cpu_to_be64(phb->mem_win_addr),

            cpu_to_be64(memory_region_size(&phb->memwindow)),

        },

    };

    uint64_t bus_reg[] = { cpu_to_be64(phb->buid), 0 };

    uint32_t interrupt_map_mask[] = {

        cpu_to_be32(b_ddddd(-1)|b_fff(0)), 0x0, 0x0, cpu_to_be32(-1)};

    uint32_t interrupt_map[PCI_SLOT_MAX * PCI_NUM_PINS][7];



    /* Start populating the FDT */

    sprintf(nodename, "pci@%" PRIx64, phb->buid);

    bus_off = fdt_add_subnode(fdt, 0, nodename);

    if (bus_off < 0) {

        return bus_off;

    }



#define _FDT(exp) \

    do { \

        int ret = (exp);                                           \

        if (ret < 0) {                                             \

            return ret;                                            \

        }                                                          \

    } while (0)



    /* Write PHB properties */

    _FDT(fdt_setprop_string(fdt, bus_off, "device_type", "pci"));

    _FDT(fdt_setprop_string(fdt, bus_off, "compatible", "IBM,Logical_PHB"));

    _FDT(fdt_setprop_cell(fdt, bus_off, "#address-cells", 0x3));

    _FDT(fdt_setprop_cell(fdt, bus_off, "#size-cells", 0x2));

    _FDT(fdt_setprop_cell(fdt, bus_off, "#interrupt-cells", 0x1));

    _FDT(fdt_setprop(fdt, bus_off, "used-by-rtas", NULL, 0));

    _FDT(fdt_setprop(fdt, bus_off, "bus-range", &bus_range, sizeof(bus_range)));

    _FDT(fdt_setprop(fdt, bus_off, "ranges", &ranges, sizeof(ranges)));

    _FDT(fdt_setprop(fdt, bus_off, "reg", &bus_reg, sizeof(bus_reg)));

    _FDT(fdt_setprop_cell(fdt, bus_off, "ibm,pci-config-space-type", 0x1));




    /* Build the interrupt-map, this must matches what is done

     * in pci_spapr_map_irq

     */

    _FDT(fdt_setprop(fdt, bus_off, "interrupt-map-mask",

                     &interrupt_map_mask, sizeof(interrupt_map_mask)));

    for (i = 0; i < PCI_SLOT_MAX; i++) {

        for (j = 0; j < PCI_NUM_PINS; j++) {

            uint32_t *irqmap = interrupt_map[i*PCI_NUM_PINS + j];

            int lsi_num = pci_spapr_swizzle(i, j);



            irqmap[0] = cpu_to_be32(b_ddddd(i)|b_fff(0));

            irqmap[1] = 0;

            irqmap[2] = 0;

            irqmap[3] = cpu_to_be32(j+1);

            irqmap[4] = cpu_to_be32(xics_phandle);

            irqmap[5] = cpu_to_be32(phb->lsi_table[lsi_num].irq);

            irqmap[6] = cpu_to_be32(0x8);

        }

    }

    /* Write interrupt map */

    _FDT(fdt_setprop(fdt, bus_off, "interrupt-map", &interrupt_map,

                     sizeof(interrupt_map)));



    object_child_foreach(OBJECT(phb), spapr_phb_children_dt,

                         &((sPAPRTCEDT){ .fdt = fdt, .node_off = bus_off }));



    return 0;

}