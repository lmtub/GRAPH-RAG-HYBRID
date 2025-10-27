int DMA_read_memory (int nchan, void *buf, int pos, int len)

{

    struct dma_regs *r = &dma_controllers[nchan > 3].regs[nchan & 3];

    target_phys_addr_t addr = ((r->pageh & 0x7f) << 24) | (r->page << 16) | r->now[ADDR];



    if (r->mode & 0x20) {

        int i;

        uint8_t *p = buf;



        cpu_physical_memory_read (addr - pos - len, buf, len);

        /* What about 16bit transfers? */

        for (i = 0; i < len >> 1; i++) {

            uint8_t b = p[len - i - 1];

            p[i] = b;

        }

    }

    else

        cpu_physical_memory_read (addr + pos, buf, len);



    return len;

}
