void OPPROTO op_store_msr (void)

{

    do_store_msr(env, T0);

    RETURN();

}
