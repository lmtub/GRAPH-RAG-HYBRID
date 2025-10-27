static int kvm_get_xcrs(CPUState *env)

{

#ifdef KVM_CAP_XCRS

    int i, ret;

    struct kvm_xcrs xcrs;



    if (!kvm_has_xcrs())

        return 0;



    ret = kvm_vcpu_ioctl(env, KVM_GET_XCRS, &xcrs);

    if (ret < 0)

        return ret;



    for (i = 0; i < xcrs.nr_xcrs; i++)

        /* Only support xcr0 now */

        if (xcrs.xcrs[0].xcr == 0) {

            env->xcr0 = xcrs.xcrs[0].value;

            break;

        }

    return 0;

#else

    return 0;

#endif

}
