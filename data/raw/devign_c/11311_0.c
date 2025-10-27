static inline void ne2000_mem_writel(NE2000State *s, uint32_t addr,

                                     uint32_t val)

{

    addr &= ~1; /* XXX: check exact behaviour if not even */

    if (addr < 32 ||

        (addr >= NE2000_PMEM_START && addr < NE2000_MEM_SIZE)) {

        stl_le_p(s->mem + addr, val);

    }

}
