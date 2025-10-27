static void gdb_vm_stopped(void *opaque, int reason)

{

    GDBState *s = opaque;

    char buf[256];

    const char *type;

    int ret;



    if (s->state == RS_SYSCALL)

        return;



    /* disable single step if it was enable */

    cpu_single_step(s->env, 0);



    if (reason == EXCP_DEBUG) {

        if (s->env->watchpoint_hit) {

            switch (s->env->watchpoint_hit->flags & BP_MEM_ACCESS) {

            case BP_MEM_READ:

                type = "r";

                break;

            case BP_MEM_ACCESS:

                type = "a";

                break;

            default:

                type = "";

                break;

            }

            snprintf(buf, sizeof(buf), "T%02x%swatch:" TARGET_FMT_lx ";",

                     SIGTRAP, type, s->env->watchpoint_hit->vaddr);

            put_packet(s, buf);

            s->env->watchpoint_hit = NULL;

            return;

        }

	tb_flush(s->env);

        ret = SIGTRAP;

    } else if (reason == EXCP_INTERRUPT) {

        ret = SIGINT;

    } else {

        ret = 0;

    }

    snprintf(buf, sizeof(buf), "S%02x", ret);

    put_packet(s, buf);

}
