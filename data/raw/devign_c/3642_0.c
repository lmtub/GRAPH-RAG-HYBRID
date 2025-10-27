static void cpu_request_exit(void *opaque, int irq, int level)

{

    CPUState *cpu = current_cpu;



    if (cpu && level) {

        cpu_exit(cpu);

    }

}
