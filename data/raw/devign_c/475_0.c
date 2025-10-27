static uint64_t bonito_cop_readl(void *opaque, target_phys_addr_t addr,

                                 unsigned size)

{

    uint32_t val;

    PCIBonitoState *s = opaque;



    val = ((uint32_t *)(&s->boncop))[addr/sizeof(uint32_t)];



    return val;

}
