static void do_dma_memory_set(dma_addr_t addr, uint8_t c, dma_addr_t len)

{

#define FILLBUF_SIZE 512

    uint8_t fillbuf[FILLBUF_SIZE];

    int l;



    memset(fillbuf, c, FILLBUF_SIZE);

    while (len > 0) {

        l = len < FILLBUF_SIZE ? len : FILLBUF_SIZE;

        cpu_physical_memory_rw(addr, fillbuf, l, true);

        len -= len;

        addr += len;

    }

}
