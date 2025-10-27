static uint32_t taihu_cpld_readb (void *opaque, hwaddr addr)

{

    taihu_cpld_t *cpld;

    uint32_t ret;



    cpld = opaque;

    switch (addr) {

    case 0x0:

        ret = cpld->reg0;

        break;

    case 0x1:

        ret = cpld->reg1;

        break;

    default:

        ret = 0;

        break;

    }



    return ret;

}
