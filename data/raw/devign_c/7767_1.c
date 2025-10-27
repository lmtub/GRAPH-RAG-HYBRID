static int check_exception(int intno, int *error_code)

{

    int first_contributory = env->old_exception == 0 ||

                              (env->old_exception >= 10 &&

                               env->old_exception <= 13);

    int second_contributory = intno == 0 ||

                               (intno >= 10 && intno <= 13);



    qemu_log_mask(CPU_LOG_INT, "check_exception old: 0x%x new 0x%x\n",

                env->old_exception, intno);



    if (env->old_exception == EXCP08_DBLE)

        cpu_abort(env, "triple fault");



    if ((first_contributory && second_contributory)

        || (env->old_exception == EXCP0E_PAGE &&

            (second_contributory || (intno == EXCP0E_PAGE)))) {

        intno = EXCP08_DBLE;

        *error_code = 0;

    }



    if (second_contributory || (intno == EXCP0E_PAGE) ||

        (intno == EXCP08_DBLE))

        env->old_exception = intno;



    return intno;

}
