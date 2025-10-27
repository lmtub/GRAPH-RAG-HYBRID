GArray *bios_linker_loader_init(void)

{

    return g_array_new(false, true /* clear */, 1);

}
