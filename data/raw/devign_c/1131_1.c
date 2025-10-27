static int handle_intercept(CPUS390XState *env)

{

    struct kvm_run *run = env->kvm_run;

    int icpt_code = run->s390_sieic.icptcode;

    int r = 0;



    dprintf("intercept: 0x%x (at 0x%lx)\n", icpt_code,

            (long)env->kvm_run->psw_addr);

    switch (icpt_code) {

        case ICPT_INSTRUCTION:

            r = handle_instruction(env, run);


        case ICPT_WAITPSW:







        case ICPT_CPU_STOP:

            if (s390_del_running_cpu(env) == 0) {





        case ICPT_SOFT_INTERCEPT:

            fprintf(stderr, "KVM unimplemented icpt SOFT\n");

            exit(1);


        case ICPT_IO:

            fprintf(stderr, "KVM unimplemented icpt IO\n");

            exit(1);


        default:

            fprintf(stderr, "Unknown intercept code: %d\n", icpt_code);

            exit(1);





    return r;
