static int ast2500_rambits(AspeedSDMCState *s)

{

    switch (s->ram_size >> 20) {

    case 128:

        return ASPEED_SDMC_AST2500_128MB;

    case 256:

        return ASPEED_SDMC_AST2500_256MB;

    case 512:

        return ASPEED_SDMC_AST2500_512MB;

    case 1024:

        return ASPEED_SDMC_AST2500_1024MB;

    default:

        break;

    }



    /* use a common default */

    error_report("warning: Invalid RAM size 0x%" PRIx64

                 ". Using default 512M", s->ram_size);

    s->ram_size = 512 << 20;

    return ASPEED_SDMC_AST2500_512MB;

}
