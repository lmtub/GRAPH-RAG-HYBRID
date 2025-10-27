static void gdb_vm_stopped(void *opaque, int reason)

{

    GDBState *s = opaque;

    char buf[256];

    int ret;



    if (s->state == RS_SYSCALL)




    /* disable single step if it was enable */

    cpu_single_step(s->env, 0);



    if (reason == EXCP_DEBUG) {








	tb_flush(s->env);

        ret = SIGTRAP;

    } else if (reason == EXCP_INTERRUPT) {

        ret = SIGINT;

    } else {

        ret = 0;


    snprintf(buf, sizeof(buf), "S%02x", ret);

