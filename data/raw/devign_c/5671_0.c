static void create_fw_cfg(const VirtBoardInfo *vbi, AddressSpace *as)

{

    hwaddr base = vbi->memmap[VIRT_FW_CFG].base;

    hwaddr size = vbi->memmap[VIRT_FW_CFG].size;

    char *nodename;



    fw_cfg_init_mem_wide(base + 8, base, 8, base + 16, as);



    nodename = g_strdup_printf("/fw-cfg@%" PRIx64, base);

    qemu_fdt_add_subnode(vbi->fdt, nodename);

    qemu_fdt_setprop_string(vbi->fdt, nodename,

                            "compatible", "qemu,fw-cfg-mmio");

    qemu_fdt_setprop_sized_cells(vbi->fdt, nodename, "reg",

                                 2, base, 2, size);

    g_free(nodename);

}
