CPUX86State *cpu_x86_init(void)

{

    CPUX86State *env;

    int i;

    static int inited;



    cpu_x86_tblocks_init();



    env = malloc(sizeof(CPUX86State));

    if (!env)

        return NULL;

    memset(env, 0, sizeof(CPUX86State));

    /* basic FPU init */

    for(i = 0;i < 8; i++)

        env->fptags[i] = 1;

    env->fpuc = 0x37f;

    /* flags setup */

    env->eflags = 0;



    /* init various static tables */

    if (!inited) {

        inited = 1;

        optimize_flags_init();

    }

    return env;

}
