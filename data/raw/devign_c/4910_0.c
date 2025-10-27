pic_read(void *opaque, target_phys_addr_t addr, unsigned int size)

{

    struct etrax_pic *fs = opaque;

    uint32_t rval;



    rval = fs->regs[addr >> 2];

    D(printf("%s %x=%x\n", __func__, addr, rval));

    return rval;

}
