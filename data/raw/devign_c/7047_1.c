void kvm_arch_pre_run(CPUState *env, struct kvm_run *run)

{

    /* Inject NMI */

    if (env->interrupt_request & CPU_INTERRUPT_NMI) {

        env->interrupt_request &= ~CPU_INTERRUPT_NMI;

        DPRINTF("injected NMI\n");

        kvm_vcpu_ioctl(env, KVM_NMI);

    }



    if (!kvm_irqchip_in_kernel()) {

        /* Force the VCPU out of its inner loop to process the INIT request */

        if (env->interrupt_request & CPU_INTERRUPT_INIT) {

            env->exit_request = 1;

        }



        /* Try to inject an interrupt if the guest can accept it */

        if (run->ready_for_interrupt_injection &&

            (env->interrupt_request & CPU_INTERRUPT_HARD) &&

            (env->eflags & IF_MASK)) {

            int irq;



            env->interrupt_request &= ~CPU_INTERRUPT_HARD;

            irq = cpu_get_pic_interrupt(env);

            if (irq >= 0) {

                struct kvm_interrupt intr;



                intr.irq = irq;

                /* FIXME: errors */

                DPRINTF("injected interrupt %d\n", irq);

                kvm_vcpu_ioctl(env, KVM_INTERRUPT, &intr);

            }

        }



        /* If we have an interrupt but the guest is not ready to receive an

         * interrupt, request an interrupt window exit.  This will

         * cause a return to userspace as soon as the guest is ready to

         * receive interrupts. */

        if ((env->interrupt_request & CPU_INTERRUPT_HARD)) {

            run->request_interrupt_window = 1;

        } else {

            run->request_interrupt_window = 0;

        }



        DPRINTF("setting tpr\n");

        run->cr8 = cpu_get_apic_tpr(env->apic_state);

    }

}
