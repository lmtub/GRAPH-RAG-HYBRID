void cpu_resume_from_signal(CPUState *env1, void *puc)

{

    env = env1;



    /* XXX: restore cpu registers saved in host registers */



    env->exception_index = -1;

    longjmp(env->jmp_env, 1);

}
