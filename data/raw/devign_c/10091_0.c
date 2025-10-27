e1000_mmio_write(void *opaque, hwaddr addr, uint64_t val,

                 unsigned size)

{

    E1000State *s = opaque;

    unsigned int index = (addr & 0x1ffff) >> 2;



    if (index < NWRITEOPS && macreg_writeops[index]) {

        macreg_writeops[index](s, index, val);

    } else if (index < NREADOPS && macreg_readops[index]) {

        DBGOUT(MMIO, "e1000_mmio_writel RO %x: 0x%04"PRIx64"\n", index<<2, val);

    } else {

        DBGOUT(UNKNOWN, "MMIO unknown write addr=0x%08x,val=0x%08"PRIx64"\n",

               index<<2, val);

    }

}
