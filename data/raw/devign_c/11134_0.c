static uint32_t sm501_palette_read(void *opaque, target_phys_addr_t addr)

{

    SM501State * s = (SM501State *)opaque;

    SM501_DPRINTF("sm501 palette read addr=%x\n", (int)addr);



    /* TODO : consider BYTE/WORD access */

    /* TODO : consider endian */



    assert(0 <= addr && addr < 0x400 * 3);

    return *(uint32_t*)&s->dc_palette[addr];

}
