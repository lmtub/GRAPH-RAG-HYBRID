void op_div (void)

{

    if (T1 != 0) {

        env->LO = (int32_t)((int32_t)T0 / (int32_t)T1);

        env->HI = (int32_t)((int32_t)T0 % (int32_t)T1);

    }

    RETURN();

}
