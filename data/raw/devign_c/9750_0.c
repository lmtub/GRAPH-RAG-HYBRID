static int spapr_populate_memory(sPAPREnvironment *spapr, void *fdt)

{

    uint32_t associativity[] = {cpu_to_be32(0x4), cpu_to_be32(0x0),

                                cpu_to_be32(0x0), cpu_to_be32(0x0),

                                cpu_to_be32(0x0)};

    char mem_name[32];

    hwaddr node0_size, mem_start;

    uint64_t mem_reg_property[2];

    int i, off;



    /* memory node(s) */

    node0_size = (nb_numa_nodes > 1) ? node_mem[0] : ram_size;



    /* RMA */

    mem_reg_property[0] = 0;

    mem_reg_property[1] = cpu_to_be64(spapr->rma_size);

    off = fdt_add_subnode(fdt, 0, "memory@0");

    _FDT(off);

    _FDT((fdt_setprop_string(fdt, off, "device_type", "memory")));

    _FDT((fdt_setprop(fdt, off, "reg", mem_reg_property,

                      sizeof(mem_reg_property))));

    _FDT((fdt_setprop(fdt, off, "ibm,associativity", associativity,

                      sizeof(associativity))));



    /* RAM: Node 0 */

    if (node0_size > spapr->rma_size) {

        mem_reg_property[0] = cpu_to_be64(spapr->rma_size);

        mem_reg_property[1] = cpu_to_be64(node0_size - spapr->rma_size);



        sprintf(mem_name, "memory@" TARGET_FMT_lx, spapr->rma_size);

        off = fdt_add_subnode(fdt, 0, mem_name);

        _FDT(off);

        _FDT((fdt_setprop_string(fdt, off, "device_type", "memory")));

        _FDT((fdt_setprop(fdt, off, "reg", mem_reg_property,

                          sizeof(mem_reg_property))));

        _FDT((fdt_setprop(fdt, off, "ibm,associativity", associativity,

                          sizeof(associativity))));

    }



    /* RAM: Node 1 and beyond */

    mem_start = node0_size;

    for (i = 1; i < nb_numa_nodes; i++) {

        mem_reg_property[0] = cpu_to_be64(mem_start);

        mem_reg_property[1] = cpu_to_be64(node_mem[i]);

        associativity[3] = associativity[4] = cpu_to_be32(i);

        sprintf(mem_name, "memory@" TARGET_FMT_lx, mem_start);

        off = fdt_add_subnode(fdt, 0, mem_name);

        _FDT(off);

        _FDT((fdt_setprop_string(fdt, off, "device_type", "memory")));

        _FDT((fdt_setprop(fdt, off, "reg", mem_reg_property,

                          sizeof(mem_reg_property))));

        _FDT((fdt_setprop(fdt, off, "ibm,associativity", associativity,

                          sizeof(associativity))));

        mem_start += node_mem[i];

    }



    return 0;

}
