static void sm501_palette_write(void *opaque,

				target_phys_addr_t addr, uint32_t value)

{

    SM501State * s = (SM501State *)opaque;

    SM501_DPRINTF("sm501 palette write addr=%x, val=%x\n",

		  (int)addr, value);



    /* TODO : consider BYTE/WORD access */

    /* TODO : consider endian */



    assert(0 <= addr && addr < 0x400 * 3);

    *(uint32_t*)&s->dc_palette[addr] = value;

}
