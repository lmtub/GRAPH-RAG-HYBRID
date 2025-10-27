static ram_addr_t qxl_rom_size(void)

{

    uint32_t required_rom_size = sizeof(QXLRom) + sizeof(QXLModes) +

                                 sizeof(qxl_modes);

    uint32_t rom_size = 8192; /* two pages */



    QEMU_BUILD_BUG_ON(required_rom_size > rom_size);

    return rom_size;

}
