static uint64_t ecc_mem_read(void *opaque, target_phys_addr_t addr,

                             unsigned size)

{

    ECCState *s = opaque;

    uint32_t ret = 0;



    switch (addr >> 2) {

    case ECC_MER:

        ret = s->regs[ECC_MER];

        trace_ecc_mem_readl_mer(ret);

        break;

    case ECC_MDR:

        ret = s->regs[ECC_MDR];

        trace_ecc_mem_readl_mdr(ret);

        break;

    case ECC_MFSR:

        ret = s->regs[ECC_MFSR];

        trace_ecc_mem_readl_mfsr(ret);

        break;

    case ECC_VCR:

        ret = s->regs[ECC_VCR];

        trace_ecc_mem_readl_vcr(ret);

        break;

    case ECC_MFAR0:

        ret = s->regs[ECC_MFAR0];

        trace_ecc_mem_readl_mfar0(ret);

        break;

    case ECC_MFAR1:

        ret = s->regs[ECC_MFAR1];

        trace_ecc_mem_readl_mfar1(ret);

        break;

    case ECC_DR:

        ret = s->regs[ECC_DR];

        trace_ecc_mem_readl_dr(ret);

        break;

    case ECC_ECR0:

        ret = s->regs[ECC_ECR0];

        trace_ecc_mem_readl_ecr0(ret);

        break;

    case ECC_ECR1:

        ret = s->regs[ECC_ECR0];

        trace_ecc_mem_readl_ecr1(ret);

        break;

    }

    return ret;

}
