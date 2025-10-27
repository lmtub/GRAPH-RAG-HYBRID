static void bonito_cop_writel(void *opaque, hwaddr addr,
                              uint64_t val, unsigned size)
{
    PCIBonitoState *s = opaque;
    ((uint32_t *)(&s->boncop))[addr/sizeof(uint32_t)] = val & 0xffffffff;