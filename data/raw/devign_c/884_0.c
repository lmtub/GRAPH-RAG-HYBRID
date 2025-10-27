static uint32_t taihu_cpld_readw (void *opaque, hwaddr addr)

{

    uint32_t ret;



    ret = taihu_cpld_readb(opaque, addr) << 8;

    ret |= taihu_cpld_readb(opaque, addr + 1);



    return ret;

}
