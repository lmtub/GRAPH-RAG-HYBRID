static void kvm_mce_inj_srar_dataload(CPUState *env, target_phys_addr_t paddr)

{

    struct kvm_x86_mce mce = {

        .bank = 9,

        .status = MCI_STATUS_VAL | MCI_STATUS_UC | MCI_STATUS_EN

                  | MCI_STATUS_MISCV | MCI_STATUS_ADDRV | MCI_STATUS_S

                  | MCI_STATUS_AR | 0x134,

        .mcg_status = MCG_STATUS_MCIP | MCG_STATUS_EIPV,

        .addr = paddr,

        .misc = (MCM_ADDR_PHYS << 6) | 0xc,

    };

    int r;



    r = kvm_set_mce(env, &mce);

    if (r < 0) {

        fprintf(stderr, "kvm_set_mce: %s\n", strerror(errno));

        abort();

    }

    kvm_mce_broadcast_rest(env);

}
