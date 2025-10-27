static void ide_identify(IDEState *s)

{

    uint16_t *p;

    unsigned int oldsize;



    memset(s->io_buffer, 0, 512);

    p = (uint16_t *)s->io_buffer;

    stw(p + 0, 0x0040);

    stw(p + 1, s->cylinders); 

    stw(p + 3, s->heads);

    stw(p + 4, 512 * s->sectors); /* sectors */

    stw(p + 5, 512); /* sector size */

    stw(p + 6, s->sectors); 

    stw(p + 20, 3); /* buffer type */

    stw(p + 21, 512); /* cache size in sectors */

    stw(p + 22, 4); /* ecc bytes */

    padstr((uint8_t *)(p + 27), "QEMU HARDDISK", 40);

#if MAX_MULT_SECTORS > 1    

    stw(p + 47, MAX_MULT_SECTORS);

#endif

    stw(p + 48, 1); /* dword I/O */

    stw(p + 49, 1 << 9); /* LBA supported, no DMA */

    stw(p + 51, 0x200); /* PIO transfer cycle */

    stw(p + 52, 0x200); /* DMA transfer cycle */

    stw(p + 54, s->cylinders);

    stw(p + 55, s->heads);

    stw(p + 56, s->sectors);

    oldsize = s->cylinders * s->heads * s->sectors;

    stw(p + 57, oldsize);

    stw(p + 58, oldsize >> 16);

    if (s->mult_sectors)

        stw(p + 59, 0x100 | s->mult_sectors);

    stw(p + 60, s->nb_sectors);

    stw(p + 61, s->nb_sectors >> 16);

    stw(p + 80, (1 << 1) | (1 << 2));

    stw(p + 82, (1 << 14));

    stw(p + 83, (1 << 14));

    stw(p + 84, (1 << 14));

    stw(p + 85, (1 << 14));

    stw(p + 86, 0);

    stw(p + 87, (1 << 14));

}
