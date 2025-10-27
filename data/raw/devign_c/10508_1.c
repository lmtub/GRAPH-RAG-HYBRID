void op_ddiv (void)

{

    if (T1 != 0) {

        env->LO = (int64_t)T0 / (int64_t)T1;

        env->HI = (int64_t)T0 % (int64_t)T1;

    }

    RETURN();

}
