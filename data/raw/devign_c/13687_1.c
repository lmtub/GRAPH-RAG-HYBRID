static void test_ivshmem_single(void)

{

    IVState state, *s;

    uint32_t data[1024];

    int i;



    setup_vm(&state);

    s = &state;



    /* valid io */

    out_reg(s, INTRMASK, 0);

    in_reg(s, INTRSTATUS);

    in_reg(s, IVPOSITION);



    out_reg(s, INTRMASK, 0xffffffff);

    g_assert_cmpuint(in_reg(s, INTRMASK), ==, 0xffffffff);

    out_reg(s, INTRSTATUS, 1);

    /* XXX: intercept IRQ, not seen in resp */

    g_assert_cmpuint(in_reg(s, INTRSTATUS), ==, 1);



    /* invalid io */

    out_reg(s, IVPOSITION, 1);

    out_reg(s, DOORBELL, 8 << 16);



    for (i = 0; i < G_N_ELEMENTS(data); i++) {

        data[i] = i;

    }

    qtest_memwrite(s->qtest, (uintptr_t)s->mem_base, data, sizeof(data));



    for (i = 0; i < G_N_ELEMENTS(data); i++) {

        g_assert_cmpuint(((uint32_t *)tmpshmem)[i], ==, i);

    }



    memset(data, 0, sizeof(data));



    qtest_memread(s->qtest, (uintptr_t)s->mem_base, data, sizeof(data));

    for (i = 0; i < G_N_ELEMENTS(data); i++) {

        g_assert_cmpuint(data[i], ==, i);

    }



    qtest_quit(s->qtest);

}
