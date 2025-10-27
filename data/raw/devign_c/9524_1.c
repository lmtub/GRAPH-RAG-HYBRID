void OPPROTO op_store_msr_32 (void)

{

    ppc_store_msr_32(env, T0);

    RETURN();

}
