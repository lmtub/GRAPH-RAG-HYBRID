void cpu_exec_init(CPUState *env)

{

    CPUState **penv;

    int cpu_index;



    if (!code_gen_ptr) {

        code_gen_ptr = code_gen_buffer;

        page_init();

        io_mem_init();

    }

    env->next_cpu = NULL;

    penv = &first_cpu;

    cpu_index = 0;

    while (*penv != NULL) {

        penv = (CPUState **)&(*penv)->next_cpu;

        cpu_index++;

    }

    env->cpu_index = cpu_index;


    *penv = env;

}