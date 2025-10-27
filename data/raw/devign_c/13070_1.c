void s390_memory_init(ram_addr_t mem_size)

{

    MemoryRegion *sysmem = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);



    /* allocate RAM for core */

    memory_region_init_ram(ram, NULL, "s390.ram", mem_size, &error_abort);

    vmstate_register_ram_global(ram);

    memory_region_add_subregion(sysmem, 0, ram);



    /* Initialize storage key device */

    s390_skeys_init();

}
