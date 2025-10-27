int qemu_devtree_setprop(void *fdt, const char *node_path,

                         const char *property, void *val_array, int size)

{

    int offset;



    offset = fdt_path_offset(fdt, node_path);

    if (offset < 0)

        return offset;



    return fdt_setprop(fdt, offset, property, val_array, size);

}
