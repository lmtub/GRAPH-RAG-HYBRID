static void memory_region_iorange_destructor(IORange *iorange)

{

    g_free(container_of(iorange, MemoryRegionIORange, iorange));

}
