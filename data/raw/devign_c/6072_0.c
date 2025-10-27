void arm_translate_init(void)

{

    cpu_env = tcg_global_reg_new(TCG_TYPE_PTR, TCG_AREG0, "env");



    cpu_T[0] = tcg_global_reg_new(TCG_TYPE_I32, TCG_AREG1, "T0");

    cpu_T[1] = tcg_global_reg_new(TCG_TYPE_I32, TCG_AREG2, "T1");

}
