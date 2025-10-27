static void gen_spr_power8_fscr(CPUPPCState *env)

{

    spr_register_kvm(env, SPR_FSCR, "FSCR",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     KVM_REG_PPC_FSCR, 0x00000000);

}
