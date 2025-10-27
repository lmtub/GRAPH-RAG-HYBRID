int rom_copy(uint8_t *dest, target_phys_addr_t addr, size_t size)

{

    target_phys_addr_t end = addr + size;

    uint8_t *s, *d = dest;

    size_t l = 0;

    Rom *rom;



    QTAILQ_FOREACH(rom, &roms, next) {

        if (rom->fw_file) {

            continue;

        }

        if (rom->addr + rom->romsize < addr)

            continue;

        if (rom->addr > end)

            break;

        if (!rom->data)

            continue;



        d = dest + (rom->addr - addr);

        s = rom->data;

        l = rom->romsize;



        if (rom->addr < addr) {

            d = dest;

            s += (addr - rom->addr);

            l -= (addr - rom->addr);

        }

        if ((d + l) > (dest + size)) {

            l = dest - d;

        }



        memcpy(d, s, l);

    }



    return (d + l) - dest;

}
