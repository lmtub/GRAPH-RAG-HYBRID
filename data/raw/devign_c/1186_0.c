static int kvm_get_xcrs(X86CPU *cpu)

{

    CPUX86State *env = &cpu->env;

    int i, ret;

    struct kvm_xcrs xcrs;



    if (!kvm_has_xcrs()) {

        return 0;

    }



    ret = kvm_vcpu_ioctl(CPU(cpu), KVM_GET_XCRS, &xcrs);

    if (ret < 0) {

        return ret;

    }



    for (i = 0; i < xcrs.nr_xcrs; i++) {

        /* Only support xcr0 now */

        if (xcrs.xcrs[i].xcr == 0) {

            env->xcr0 = xcrs.xcrs[i].value;

            break;

        }

    }

    return 0;

}
