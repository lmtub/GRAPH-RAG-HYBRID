static void rc4030_dma_tt_update(rc4030State *s, uint32_t new_tl_base,

                                 uint32_t new_tl_limit)

{

    int entries, i;

    dma_pagetable_entry *dma_tl_contents;



    if (s->dma_tl_limit) {

        /* write old dma tl table to physical memory */

        memory_region_del_subregion(get_system_memory(), &s->dma_tt_alias);

        cpu_physical_memory_write(s->dma_tl_limit & 0x7fffffff,

                                  memory_region_get_ram_ptr(&s->dma_tt),

                                  memory_region_size(&s->dma_tt_alias));

    }

    object_unparent(OBJECT(&s->dma_tt_alias));



    s->dma_tl_base = new_tl_base;

    s->dma_tl_limit = new_tl_limit;

    new_tl_base &= 0x7fffffff;



    if (s->dma_tl_limit) {

        uint64_t dma_tt_size;

        if (s->dma_tl_limit <= memory_region_size(&s->dma_tt)) {

            dma_tt_size = s->dma_tl_limit;

        } else {

            dma_tt_size = memory_region_size(&s->dma_tt);

        }

        memory_region_init_alias(&s->dma_tt_alias, OBJECT(s),

                                 "dma-table-alias",

                                 &s->dma_tt, 0, dma_tt_size);

        dma_tl_contents = memory_region_get_ram_ptr(&s->dma_tt);

        cpu_physical_memory_read(new_tl_base, dma_tl_contents, dma_tt_size);



        memory_region_transaction_begin();

        entries = dma_tt_size / sizeof(dma_pagetable_entry);

        for (i = 0; i < entries; i++) {

            rc4030_dma_as_update_one(s, i, dma_tl_contents[i].frame);

        }

        memory_region_add_subregion(get_system_memory(), new_tl_base,

                                    &s->dma_tt_alias);

        memory_region_transaction_commit();

    } else {

        memory_region_init(&s->dma_tt_alias, OBJECT(s),

                           "dma-table-alias", 0);

    }

}
