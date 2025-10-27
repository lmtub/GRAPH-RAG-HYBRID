static void openrisc_pic_cpu_handler(void *opaque, int irq, int level)

{

    OpenRISCCPU *cpu = (OpenRISCCPU *)opaque;

    CPUState *cs = CPU(cpu);

    int i;

    uint32_t irq_bit = 1 << irq;



    if (irq > 31 || irq < 0) {

        return;

    }



    if (level) {

        cpu->env.picsr |= irq_bit;

    } else {

        cpu->env.picsr &= ~irq_bit;

    }



    for (i = 0; i < 32; i++) {

        if ((cpu->env.picsr && (1 << i)) && (cpu->env.picmr && (1 << i))) {

            cpu_interrupt(cs, CPU_INTERRUPT_HARD);

        } else {

            cpu_reset_interrupt(cs, CPU_INTERRUPT_HARD);

            cpu->env.picsr &= ~(1 << i);

        }

    }

}
