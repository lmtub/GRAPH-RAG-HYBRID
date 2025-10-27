static void ecc_mem_write(void *opaque, target_phys_addr_t addr, uint64_t val,

                          unsigned size)

{

    ECCState *s = opaque;



    switch (addr >> 2) {

    case ECC_MER:

        if (s->version == ECC_MCC)

            s->regs[ECC_MER] = (val & ECC_MER_MASK_0);

        else if (s->version == ECC_EMC)

            s->regs[ECC_MER] = s->version | (val & ECC_MER_MASK_1);

        else if (s->version == ECC_SMC)

            s->regs[ECC_MER] = s->version | (val & ECC_MER_MASK_2);

        trace_ecc_mem_writel_mer(val);

        break;

    case ECC_MDR:

        s->regs[ECC_MDR] =  val & ECC_MDR_MASK;

        trace_ecc_mem_writel_mdr(val);

        break;

    case ECC_MFSR:

        s->regs[ECC_MFSR] =  val;

        qemu_irq_lower(s->irq);

        trace_ecc_mem_writel_mfsr(val);

        break;

    case ECC_VCR:

        s->regs[ECC_VCR] =  val;

        trace_ecc_mem_writel_vcr(val);

        break;

    case ECC_DR:

        s->regs[ECC_DR] =  val;

        trace_ecc_mem_writel_dr(val);

        break;

    case ECC_ECR0:

        s->regs[ECC_ECR0] =  val;

        trace_ecc_mem_writel_ecr0(val);

        break;

    case ECC_ECR1:

        s->regs[ECC_ECR0] =  val;

        trace_ecc_mem_writel_ecr1(val);

        break;

    }

}
