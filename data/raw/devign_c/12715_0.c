static inline int target_rt_restore_ucontext(CPUM68KState *env,

                                             struct target_ucontext *uc,

                                             int *pd0)

{

    int temp;

    target_greg_t *gregs = uc->tuc_mcontext.gregs;

    

    __get_user(temp, &uc->tuc_mcontext.version);

    if (temp != TARGET_MCONTEXT_VERSION)

        goto badframe;



    /* restore passed registers */

    __get_user(env->dregs[0], &gregs[0]);

    __get_user(env->dregs[1], &gregs[1]);

    __get_user(env->dregs[2], &gregs[2]);

    __get_user(env->dregs[3], &gregs[3]);

    __get_user(env->dregs[4], &gregs[4]);

    __get_user(env->dregs[5], &gregs[5]);

    __get_user(env->dregs[6], &gregs[6]);

    __get_user(env->dregs[7], &gregs[7]);

    __get_user(env->aregs[0], &gregs[8]);

    __get_user(env->aregs[1], &gregs[9]);

    __get_user(env->aregs[2], &gregs[10]);

    __get_user(env->aregs[3], &gregs[11]);

    __get_user(env->aregs[4], &gregs[12]);

    __get_user(env->aregs[5], &gregs[13]);

    __get_user(env->aregs[6], &gregs[14]);

    __get_user(env->aregs[7], &gregs[15]);

    __get_user(env->pc, &gregs[16]);

    __get_user(temp, &gregs[17]);

    env->sr = (env->sr & 0xff00) | (temp & 0xff);



    *pd0 = env->dregs[0];

    return 0;



badframe:

    return 1;

}
