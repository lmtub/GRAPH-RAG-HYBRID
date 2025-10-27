static void *atomic_mmu_lookup(CPUArchState *env, target_ulong addr,

                               TCGMemOpIdx oi, uintptr_t retaddr)

{

    size_t mmu_idx = get_mmuidx(oi);

    size_t index = (addr >> TARGET_PAGE_BITS) & (CPU_TLB_SIZE - 1);

    CPUTLBEntry *tlbe = &env->tlb_table[mmu_idx][index];

    target_ulong tlb_addr = tlbe->addr_write;

    TCGMemOp mop = get_memop(oi);

    int a_bits = get_alignment_bits(mop);

    int s_bits = mop & MO_SIZE;



    /* Adjust the given return address.  */

    retaddr -= GETPC_ADJ;



    /* Enforce guest required alignment.  */

    if (unlikely(a_bits > 0 && (addr & ((1 << a_bits) - 1)))) {

        /* ??? Maybe indicate atomic op to cpu_unaligned_access */

        cpu_unaligned_access(ENV_GET_CPU(env), addr, MMU_DATA_STORE,

                             mmu_idx, retaddr);

    }



    /* Enforce qemu required alignment.  */

    if (unlikely(addr & ((1 << s_bits) - 1))) {

        /* We get here if guest alignment was not requested,

           or was not enforced by cpu_unaligned_access above.

           We might widen the access and emulate, but for now

           mark an exception and exit the cpu loop.  */

        goto stop_the_world;

    }



    /* Check TLB entry and enforce page permissions.  */

    if ((addr & TARGET_PAGE_MASK)

        != (tlb_addr & (TARGET_PAGE_MASK | TLB_INVALID_MASK))) {

        if (!VICTIM_TLB_HIT(addr_write, addr)) {

            tlb_fill(ENV_GET_CPU(env), addr, MMU_DATA_STORE, mmu_idx, retaddr);

        }

        tlb_addr = tlbe->addr_write;

    }



    /* Check notdirty */

    if (unlikely(tlb_addr & TLB_NOTDIRTY)) {

        tlb_set_dirty(ENV_GET_CPU(env), addr);

        tlb_addr = tlb_addr & ~TLB_NOTDIRTY;

    }



    /* Notice an IO access  */

    if (unlikely(tlb_addr & ~TARGET_PAGE_MASK)) {

        /* There's really nothing that can be done to

           support this apart from stop-the-world.  */

        goto stop_the_world;

    }



    /* Let the guest notice RMW on a write-only page.  */

    if (unlikely(tlbe->addr_read != tlb_addr)) {

        tlb_fill(ENV_GET_CPU(env), addr, MMU_DATA_LOAD, mmu_idx, retaddr);

        /* Since we don't support reads and writes to different addresses,

           and we do have the proper page loaded for write, this shouldn't

           ever return.  But just in case, handle via stop-the-world.  */

        goto stop_the_world;

    }



    return (void *)((uintptr_t)addr + tlbe->addend);



 stop_the_world:

    cpu_loop_exit_atomic(ENV_GET_CPU(env), retaddr);

}
