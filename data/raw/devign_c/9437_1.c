static void ppc_cpu_realizefn(DeviceState *dev, Error **errp)
{
    CPUState *cs = CPU(dev);
    PowerPCCPU *cpu = POWERPC_CPU(dev);
    PowerPCCPUClass *pcc = POWERPC_CPU_GET_CLASS(cpu);
    Error *local_err = NULL;
#if !defined(CONFIG_USER_ONLY)
    int max_smt = kvm_enabled() ? kvmppc_smt_threads() : 1;
#endif
#if !defined(CONFIG_USER_ONLY)
    if (smp_threads > max_smt) {
        error_setg(errp, "Cannot support more than %d threads on PPC with %s",
                   max_smt, kvm_enabled() ? "KVM" : "TCG");
    if (!is_power_of_2(smp_threads)) {
        error_setg(errp, "Cannot support %d threads on PPC with %s, "
                   "threads count must be a power of 2.",
                   smp_threads, kvm_enabled() ? "KVM" : "TCG");
#endif
    cpu_exec_init(cs, &local_err);
    if (local_err != NULL) {
        error_propagate(errp, local_err);
#if !defined(CONFIG_USER_ONLY)
    cpu->cpu_dt_id = (cs->cpu_index / smp_threads) * max_smt
        + (cs->cpu_index % smp_threads);
#endif
    if (tcg_enabled()) {
        if (ppc_fixup_cpu(cpu) != 0) {
            error_setg(errp, "Unable to emulate selected CPU with TCG");
#if defined(TARGET_PPCEMB)
    if (!ppc_cpu_is_valid(pcc)) {
        error_setg(errp, "CPU does not possess a BookE or 4xx MMU. "
                   "Please use qemu-system-ppc or qemu-system-ppc64 instead "
                   "or choose another CPU model.");
#endif
    create_ppc_opcodes(cpu, &local_err);
    if (local_err != NULL) {
        error_propagate(errp, local_err);
    init_ppc_proc(cpu);
    if (pcc->insns_flags & PPC_FLOAT) {
        gdb_register_coprocessor(cs, gdb_get_float_reg, gdb_set_float_reg,
                                 33, "power-fpu.xml", 0);
    if (pcc->insns_flags & PPC_ALTIVEC) {
        gdb_register_coprocessor(cs, gdb_get_avr_reg, gdb_set_avr_reg,
                                 34, "power-altivec.xml", 0);
    if (pcc->insns_flags & PPC_SPE) {
        gdb_register_coprocessor(cs, gdb_get_spe_reg, gdb_set_spe_reg,
                                 34, "power-spe.xml", 0);
    if (pcc->insns_flags2 & PPC2_VSX) {
        gdb_register_coprocessor(cs, gdb_get_vsx_reg, gdb_set_vsx_reg,
                                 32, "power-vsx.xml", 0);
    qemu_init_vcpu(cs);
    pcc->parent_realize(dev, errp);
#if defined(PPC_DUMP_CPU)
    {
        CPUPPCState *env = &cpu->env;
        const char *mmu_model, *excp_model, *bus_model;
        switch (env->mmu_model) {
        case POWERPC_MMU_32B:
            mmu_model = "PowerPC 32";
            break;
        case POWERPC_MMU_SOFT_6xx:
            mmu_model = "PowerPC 6xx/7xx with software driven TLBs";
            break;
        case POWERPC_MMU_SOFT_74xx:
            mmu_model = "PowerPC 74xx with software driven TLBs";
            break;
        case POWERPC_MMU_SOFT_4xx:
            mmu_model = "PowerPC 4xx with software driven TLBs";
            break;
        case POWERPC_MMU_SOFT_4xx_Z:
            mmu_model = "PowerPC 4xx with software driven TLBs "
                "and zones protections";
            break;
        case POWERPC_MMU_REAL:
            mmu_model = "PowerPC real mode only";
            break;
        case POWERPC_MMU_MPC8xx:
            mmu_model = "PowerPC MPC8xx";
            break;
        case POWERPC_MMU_BOOKE:
            mmu_model = "PowerPC BookE";
            break;
        case POWERPC_MMU_BOOKE206:
            mmu_model = "PowerPC BookE 2.06";
            break;
        case POWERPC_MMU_601:
            mmu_model = "PowerPC 601";
            break;
#if defined (TARGET_PPC64)
        case POWERPC_MMU_64B:
            mmu_model = "PowerPC 64";
            break;
#endif
        default:
            mmu_model = "Unknown or invalid";
            break;
        switch (env->excp_model) {
        case POWERPC_EXCP_STD:
            excp_model = "PowerPC";
            break;
        case POWERPC_EXCP_40x:
            excp_model = "PowerPC 40x";
            break;
        case POWERPC_EXCP_601:
            excp_model = "PowerPC 601";
            break;
        case POWERPC_EXCP_602:
            excp_model = "PowerPC 602";
            break;
        case POWERPC_EXCP_603:
            excp_model = "PowerPC 603";
            break;
        case POWERPC_EXCP_603E:
            excp_model = "PowerPC 603e";
            break;
        case POWERPC_EXCP_604:
            excp_model = "PowerPC 604";
            break;
        case POWERPC_EXCP_7x0:
            excp_model = "PowerPC 740/750";
            break;
        case POWERPC_EXCP_7x5:
            excp_model = "PowerPC 745/755";
            break;
        case POWERPC_EXCP_74xx:
            excp_model = "PowerPC 74xx";
            break;
        case POWERPC_EXCP_BOOKE:
            excp_model = "PowerPC BookE";
            break;
#if defined (TARGET_PPC64)
        case POWERPC_EXCP_970:
            excp_model = "PowerPC 970";
            break;
#endif
        default:
            excp_model = "Unknown or invalid";
            break;
        switch (env->bus_model) {
        case PPC_FLAGS_INPUT_6xx:
            bus_model = "PowerPC 6xx";
            break;
        case PPC_FLAGS_INPUT_BookE:
            bus_model = "PowerPC BookE";
            break;
        case PPC_FLAGS_INPUT_405:
            bus_model = "PowerPC 405";
            break;
        case PPC_FLAGS_INPUT_401:
            bus_model = "PowerPC 401/403";
            break;
        case PPC_FLAGS_INPUT_RCPU:
            bus_model = "RCPU / MPC8xx";
            break;
#if defined (TARGET_PPC64)
        case PPC_FLAGS_INPUT_970:
            bus_model = "PowerPC 970";
            break;
#endif
        default:
            bus_model = "Unknown or invalid";
            break;
        printf("PowerPC %-12s : PVR %08x MSR %016" PRIx64 "\n"
               "    MMU model        : %s\n",
               object_class_get_name(OBJECT_CLASS(pcc)),
               pcc->pvr, pcc->msr_mask, mmu_model);
#if !defined(CONFIG_USER_ONLY)
        if (env->tlb.tlb6) {
            printf("                       %d %s TLB in %d ways\n",
                   env->nb_tlb, env->id_tlbs ? "splitted" : "merged",
                   env->nb_ways);
#endif
        printf("    Exceptions model : %s\n"
               "    Bus model        : %s\n",
               excp_model, bus_model);
        printf("    MSR features     :\n");
        if (env->flags & POWERPC_FLAG_SPE)
            printf("                        signal processing engine enable"
                   "\n");
        else if (env->flags & POWERPC_FLAG_VRE)
            printf("                        vector processor enable\n");
        if (env->flags & POWERPC_FLAG_TGPR)
            printf("                        temporary GPRs\n");
        else if (env->flags & POWERPC_FLAG_CE)
            printf("                        critical input enable\n");
        if (env->flags & POWERPC_FLAG_SE)
            printf("                        single-step trace mode\n");
        else if (env->flags & POWERPC_FLAG_DWE)
            printf("                        debug wait enable\n");
        else if (env->flags & POWERPC_FLAG_UBLE)
            printf("                        user BTB lock enable\n");
        if (env->flags & POWERPC_FLAG_BE)
            printf("                        branch-step trace mode\n");
        else if (env->flags & POWERPC_FLAG_DE)
            printf("                        debug interrupt enable\n");
        if (env->flags & POWERPC_FLAG_PX)
            printf("                        inclusive protection\n");
        else if (env->flags & POWERPC_FLAG_PMM)
            printf("                        performance monitor mark\n");
        if (env->flags == POWERPC_FLAG_NONE)
            printf("                        none\n");
        printf("    Time-base/decrementer clock source: %s\n",
               env->flags & POWERPC_FLAG_RTC_CLK ? "RTC clock" : "bus clock");
        dump_ppc_insns(env);
        dump_ppc_sprs(env);
        fflush(stdout);
#endif