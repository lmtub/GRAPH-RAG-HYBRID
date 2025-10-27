int qemu_devtree_setprop_cell(void *fdt, const char *node_path,

                              const char *property, uint32_t val)

{

    int offset;



    offset = fdt_path_offset(fdt, node_path);

    if (offset < 0)

        return offset;



    return fdt_setprop_cell(fdt, offset, property, val);

}
