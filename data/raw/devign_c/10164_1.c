static void vapic_map_rom_writable(VAPICROMState *s)

{

    hwaddr rom_paddr = s->rom_state_paddr & ROM_BLOCK_MASK;

    MemoryRegionSection section;

    MemoryRegion *as;

    size_t rom_size;

    uint8_t *ram;



    as = sysbus_address_space(&s->busdev);



    if (s->rom_mapped_writable) {

        memory_region_del_subregion(as, &s->rom);

        memory_region_destroy(&s->rom);

    }



    /* grab RAM memory region (region @rom_paddr may still be pc.rom) */

    section = memory_region_find(as, 0, 1);



    /* read ROM size from RAM region */

    ram = memory_region_get_ram_ptr(section.mr);

    rom_size = ram[rom_paddr + 2] * ROM_BLOCK_SIZE;

    s->rom_size = rom_size;



    /* We need to round to avoid creating subpages

     * from which we cannot run code. */

    rom_size += rom_paddr & ~TARGET_PAGE_MASK;

    rom_paddr &= TARGET_PAGE_MASK;

    rom_size = TARGET_PAGE_ALIGN(rom_size);



    memory_region_init_alias(&s->rom, OBJECT(s), "kvmvapic-rom", section.mr,

                             rom_paddr, rom_size);

    memory_region_add_subregion_overlap(as, rom_paddr, &s->rom, 1000);

    s->rom_mapped_writable = true;

    memory_region_unref(section.mr);

}
