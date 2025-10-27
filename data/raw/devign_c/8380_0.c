static uint64_t nand_read(void *opaque, target_phys_addr_t addr, unsigned size)

{

    struct nand_state_t *s = opaque;

    uint32_t r;

    int rdy;



    r = nand_getio(s->nand);

    nand_getpins(s->nand, &rdy);

    s->rdy = rdy;



    DNAND(printf("%s addr=%x r=%x\n", __func__, addr, r));

    return r;

}
