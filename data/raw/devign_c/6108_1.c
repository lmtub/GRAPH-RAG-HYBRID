static const void *boston_fdt_filter(void *opaque, const void *fdt_orig,

                                     const void *match_data, hwaddr *load_addr)

{

    BostonState *s = BOSTON(opaque);

    MachineState *machine = s->mach;

    const char *cmdline;

    int err;

    void *fdt;

    size_t fdt_sz, ram_low_sz, ram_high_sz;



    fdt_sz = fdt_totalsize(fdt_orig) * 2;

    fdt = g_malloc0(fdt_sz);



    err = fdt_open_into(fdt_orig, fdt, fdt_sz);

    if (err) {

        fprintf(stderr, "unable to open FDT\n");

        return NULL;

    }



    cmdline = (machine->kernel_cmdline && machine->kernel_cmdline[0])

            ? machine->kernel_cmdline : " ";

    err = qemu_fdt_setprop_string(fdt, "/chosen", "bootargs", cmdline);

    if (err < 0) {

        fprintf(stderr, "couldn't set /chosen/bootargs\n");

        return NULL;

    }



    ram_low_sz = MIN(256 * M_BYTE, machine->ram_size);

    ram_high_sz = machine->ram_size - ram_low_sz;

    qemu_fdt_setprop_sized_cells(fdt, "/memory@0", "reg",

                                 1, 0x00000000, 1, ram_low_sz,

                                 1, 0x90000000, 1, ram_high_sz);



    fdt = g_realloc(fdt, fdt_totalsize(fdt));

    qemu_fdt_dumpdtb(fdt, fdt_sz);



    s->fdt_base = *load_addr;



    return fdt;

}
