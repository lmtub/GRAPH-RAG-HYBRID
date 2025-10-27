static Rom *find_rom(target_phys_addr_t addr)

{

    Rom *rom;



    QTAILQ_FOREACH(rom, &roms, next) {

        if (rom->fw_file) {

            continue;

        }

        if (rom->addr > addr) {

            continue;

        }

        if (rom->addr + rom->romsize < addr) {

            continue;

        }

        return rom;

    }

    return NULL;

}
