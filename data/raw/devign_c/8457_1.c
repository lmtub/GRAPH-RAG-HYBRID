static void qemu_tcg_init_vcpu(CPUState *cpu)

{

    char thread_name[VCPU_THREAD_NAME_SIZE];

    static QemuCond *tcg_halt_cond;

    static QemuThread *tcg_cpu_thread;



    /* share a single thread for all cpus with TCG */

    if (!tcg_cpu_thread) {

        cpu->thread = g_malloc0(sizeof(QemuThread));

        cpu->halt_cond = g_malloc0(sizeof(QemuCond));

        qemu_cond_init(cpu->halt_cond);

        tcg_halt_cond = cpu->halt_cond;

        snprintf(thread_name, VCPU_THREAD_NAME_SIZE, "CPU %d/TCG",

                 cpu->cpu_index);

        qemu_thread_create(cpu->thread, thread_name, qemu_tcg_cpu_thread_fn,

                           cpu, QEMU_THREAD_JOINABLE);

#ifdef _WIN32

        cpu->hThread = qemu_thread_get_handle(cpu->thread);

#endif

        while (!cpu->created) {

            qemu_cond_wait(&qemu_cpu_cond, &qemu_global_mutex);

        }

        tcg_cpu_thread = cpu->thread;

    } else {

        cpu->thread = tcg_cpu_thread;

        cpu->halt_cond = tcg_halt_cond;

    }

}
