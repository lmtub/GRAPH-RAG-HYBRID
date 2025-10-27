static void taihu_cpld_writew (void *opaque,

                               hwaddr addr, uint32_t value)

{

    taihu_cpld_writeb(opaque, addr, (value >> 8) & 0xFF);

    taihu_cpld_writeb(opaque, addr + 1, value & 0xFF);

}
