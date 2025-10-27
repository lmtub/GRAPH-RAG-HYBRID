void tb_phys_invalidate(TranslationBlock *tb, tb_page_addr_t page_addr)

{

    CPUState *cpu;

    PageDesc *p;

    uint32_t h;

    tb_page_addr_t phys_pc;



    assert_tb_locked();



    atomic_set(&tb->cflags, tb->cflags | CF_INVALID);



    /* remove the TB from the hash list */

    phys_pc = tb->page_addr[0] + (tb->pc & ~TARGET_PAGE_MASK);

    h = tb_hash_func(phys_pc, tb->pc, tb->flags, tb->cflags & CF_HASH_MASK,

                     tb->trace_vcpu_dstate);

    qht_remove(&tb_ctx.htable, tb, h);



    /* remove the TB from the page list */

    if (tb->page_addr[0] != page_addr) {

        p = page_find(tb->page_addr[0] >> TARGET_PAGE_BITS);

        tb_page_remove(&p->first_tb, tb);

        invalidate_page_bitmap(p);

    }

    if (tb->page_addr[1] != -1 && tb->page_addr[1] != page_addr) {

        p = page_find(tb->page_addr[1] >> TARGET_PAGE_BITS);

        tb_page_remove(&p->first_tb, tb);

        invalidate_page_bitmap(p);

    }



    /* remove the TB from the hash list */

    h = tb_jmp_cache_hash_func(tb->pc);

    CPU_FOREACH(cpu) {

        if (atomic_read(&cpu->tb_jmp_cache[h]) == tb) {

            atomic_set(&cpu->tb_jmp_cache[h], NULL);

        }

    }



    /* suppress this TB from the two jump lists */

    tb_remove_from_jmp_list(tb, 0);

    tb_remove_from_jmp_list(tb, 1);



    /* suppress any remaining jumps to this TB */

    tb_jmp_unlink(tb);



    tb_ctx.tb_phys_invalidate_count++;

}
