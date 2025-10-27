static void eeprom_generate(eeprom24c0x_t *eeprom, ram_addr_t ram_size)

{

    enum { SDR = 0x4, DDR2 = 0x8 } type;

    uint8_t *spd = eeprom->contents;

    uint8_t nbanks = 0;

    uint16_t density = 0;

    int i;



    /* work in terms of MB */

    ram_size >>= 20;



    while ((ram_size >= 4) && (nbanks <= 2)) {

        int sz_log2 = MIN(31 - clz32(ram_size), 14);

        nbanks++;

        density |= 1 << (sz_log2 - 2);

        ram_size -= 1 << sz_log2;

    }



    /* split to 2 banks if possible */

    if ((nbanks == 1) && (density > 1)) {

        nbanks++;

        density >>= 1;

    }



    if (density & 0xff00) {

        density = (density & 0xe0) | ((density >> 8) & 0x1f);

        type = DDR2;

    } else if (!(density & 0x1f)) {

        type = DDR2;

    } else {

        type = SDR;

    }



    if (ram_size) {

        fprintf(stderr, "Warning: SPD cannot represent final %dMB"

                " of SDRAM\n", (int)ram_size);

    }



    /* fill in SPD memory information */

    spd[2] = type;

    spd[5] = nbanks;

    spd[31] = density;



    /* checksum */

    spd[63] = 0;

    for (i = 0; i < 63; i++) {

        spd[63] += spd[i];

    }

}
