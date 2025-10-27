static void tcg_cpu_exec(void)

{

    int ret = 0;



    if (next_cpu == NULL)

        next_cpu = first_cpu;

    for (; next_cpu != NULL; next_cpu = next_cpu->next_cpu) {

        CPUState *env = cur_cpu = next_cpu;



        if (timer_alarm_pending) {

            timer_alarm_pending = 0;

            break;

        }

        if (cpu_can_run(env))

            ret = qemu_cpu_exec(env);

        else if (env->stop)

            break;



        if (ret == EXCP_DEBUG) {

            gdb_set_stop_cpu(env);

            debug_requested = 1;

            break;

        }

    }

}
