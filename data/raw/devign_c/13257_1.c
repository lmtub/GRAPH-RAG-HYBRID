void arm_cpu_do_transaction_failed(CPUState *cs, hwaddr physaddr,

                                   vaddr addr, unsigned size,

                                   MMUAccessType access_type,

                                   int mmu_idx, MemTxAttrs attrs,

                                   MemTxResult response, uintptr_t retaddr)

{

    ARMCPU *cpu = ARM_CPU(cs);

    ARMMMUFaultInfo fi = {};



    /* now we have a real cpu fault */

    cpu_restore_state(cs, retaddr);



    /* The EA bit in syndromes and fault status registers is an

     * IMPDEF classification of external aborts. ARM implementations

     * usually use this to indicate AXI bus Decode error (0) or

     * Slave error (1); in QEMU we follow that.

     */

    fi.ea = (response != MEMTX_DECODE_ERROR);

    fi.type = ARMFault_SyncExternal;

    deliver_fault(cpu, addr, access_type, mmu_idx, &fi);

}
