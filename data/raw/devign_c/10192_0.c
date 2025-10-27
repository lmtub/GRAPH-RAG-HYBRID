int kvmppc_get_hypercall(CPUPPCState *env, uint8_t *buf, int buf_len)

{

    PowerPCCPU *cpu = ppc_env_get_cpu(env);

    CPUState *cs = CPU(cpu);

    uint32_t *hc = (uint32_t*)buf;



    struct kvm_ppc_pvinfo pvinfo;



    if (kvm_check_extension(cs->kvm_state, KVM_CAP_PPC_GET_PVINFO) &&

        !kvm_vm_ioctl(cs->kvm_state, KVM_PPC_GET_PVINFO, &pvinfo)) {

        memcpy(buf, pvinfo.hcall, buf_len);



        return 0;

    }



    /*

     * Fallback to always fail hypercalls:

     *

     *     li r3, -1

     *     nop

     *     nop

     *     nop

     */



    hc[0] = 0x3860ffff;

    hc[1] = 0x60000000;

    hc[2] = 0x60000000;

    hc[3] = 0x60000000;



    return 0;

}
