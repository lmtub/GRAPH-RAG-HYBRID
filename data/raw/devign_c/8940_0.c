static void taihu_cpld_writel (void *opaque,

                               hwaddr addr, uint32_t value)

{

    taihu_cpld_writel(opaque, addr, (value >> 24) & 0xFF);

    taihu_cpld_writel(opaque, addr + 1, (value >> 16) & 0xFF);

    taihu_cpld_writel(opaque, addr + 2, (value >> 8) & 0xFF);

    taihu_cpld_writeb(opaque, addr + 3, value & 0xFF);

}
