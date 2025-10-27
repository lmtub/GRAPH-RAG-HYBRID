uint32_t qemu_devtree_alloc_phandle(void *fdt)

{

    static int phandle = 0x8000;



    return phandle++;

}
