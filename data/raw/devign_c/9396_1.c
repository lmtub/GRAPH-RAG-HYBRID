static uint64_t arm_ldq_ptw(CPUState *cs, hwaddr addr, bool is_secure,

                            ARMMMUIdx mmu_idx, ARMMMUFaultInfo *fi)

{

    ARMCPU *cpu = ARM_CPU(cs);

    CPUARMState *env = &cpu->env;

    MemTxAttrs attrs = {};

    AddressSpace *as;



    attrs.secure = is_secure;

    as = arm_addressspace(cs, attrs);

    addr = S1_ptw_translate(env, mmu_idx, addr, attrs, fi);

    if (fi->s1ptw) {

        return 0;

    }

    if (regime_translation_big_endian(env, mmu_idx)) {

        return address_space_ldq_be(as, addr, attrs, NULL);

    } else {

        return address_space_ldq_le(as, addr, attrs, NULL);

    }

}
