void qemu_cpu_kick(void *_env)

{

    CPUState *env = _env;



    qemu_cond_broadcast(env->halt_cond);

    if (!env->thread_kicked) {

        qemu_cpu_kick_thread(env);

        env->thread_kicked = true;

    }

}
