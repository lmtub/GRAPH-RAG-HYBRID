static int kvm_put_vcpu_events(X86CPU *cpu, int level)

{

    CPUX86State *env = &cpu->env;

    struct kvm_vcpu_events events;



    if (!kvm_has_vcpu_events()) {

        return 0;

    }



    events.exception.injected = (env->exception_injected >= 0);

    events.exception.nr = env->exception_injected;

    events.exception.has_error_code = env->has_error_code;

    events.exception.error_code = env->error_code;

    events.exception.pad = 0;



    events.interrupt.injected = (env->interrupt_injected >= 0);

    events.interrupt.nr = env->interrupt_injected;

    events.interrupt.soft = env->soft_interrupt;



    events.nmi.injected = env->nmi_injected;

    events.nmi.pending = env->nmi_pending;

    events.nmi.masked = !!(env->hflags2 & HF2_NMI_MASK);

    events.nmi.pad = 0;



    events.sipi_vector = env->sipi_vector;



    events.flags = 0;

    if (level >= KVM_PUT_RESET_STATE) {

        events.flags |=

            KVM_VCPUEVENT_VALID_NMI_PENDING | KVM_VCPUEVENT_VALID_SIPI_VECTOR;

    }



    return kvm_vcpu_ioctl(CPU(cpu), KVM_SET_VCPU_EVENTS, &events);

}
