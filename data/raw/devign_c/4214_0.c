void op_mfc0_ebase (void)

{

    T0 = (int32_t)env->CP0_EBase;

    RETURN();

}
