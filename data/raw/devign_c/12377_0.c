static void kvm_mce_inj_srao_memscrub2(CPUState *env, target_phys_addr_t paddr)

{

    struct kvm_x86_mce mce = {

        .bank = 9,

        .status = MCI_STATUS_VAL | MCI_STATUS_UC | MCI_STATUS_EN

                  | MCI_STATUS_MISCV | MCI_STATUS_ADDRV | MCI_STATUS_S

                  | 0xc0,

        .mcg_status = MCG_STATUS_MCIP | MCG_STATUS_RIPV,

        .addr = paddr,

        .misc = (MCM_ADDR_PHYS << 6) | 0xc,

    };



    kvm_inject_x86_mce_on(env, &mce, ABORT_ON_ERROR);

    kvm_mce_broadcast_rest(env);

}
