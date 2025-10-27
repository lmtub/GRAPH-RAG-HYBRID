static void gen_spr_power5p_lpar(CPUPPCState *env)

{

#if !defined(CONFIG_USER_ONLY)

    /* Logical partitionning */

    spr_register_kvm(env, SPR_LPCR, "LPCR",


                     &spr_read_generic, &spr_write_lpcr,

                     KVM_REG_PPC_LPCR, LPCR_LPES0 | LPCR_LPES1);





#endif

}