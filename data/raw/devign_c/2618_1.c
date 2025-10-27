int qemu_devtree_nop_node(void *fdt, const char *node_path)

{

    int offset;



    offset = fdt_path_offset(fdt, node_path);

    if (offset < 0)

        return offset;



    return fdt_nop_node(fdt, offset);

}
