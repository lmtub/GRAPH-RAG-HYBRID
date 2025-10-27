void test_segs(void)

{

    struct modify_ldt_ldt_s ldt;

    long long ldt_table[3];

    int res, res2;

    char tmp;

    struct {

        uint32_t offset;

        uint16_t seg;

    } __attribute__((packed)) segoff;



    ldt.entry_number = 1;

    ldt.base_addr = (unsigned long)&seg_data1;

    ldt.limit = (sizeof(seg_data1) + 0xfff) >> 12;

    ldt.seg_32bit = 1;

    ldt.contents = MODIFY_LDT_CONTENTS_DATA;

    ldt.read_exec_only = 0;

    ldt.limit_in_pages = 1;

    ldt.seg_not_present = 0;

    ldt.useable = 1;

    modify_ldt(1, &ldt, sizeof(ldt)); /* write ldt entry */



    ldt.entry_number = 2;

    ldt.base_addr = (unsigned long)&seg_data2;

    ldt.limit = (sizeof(seg_data2) + 0xfff) >> 12;

    ldt.seg_32bit = 1;

    ldt.contents = MODIFY_LDT_CONTENTS_DATA;

    ldt.read_exec_only = 0;

    ldt.limit_in_pages = 1;

    ldt.seg_not_present = 0;

    ldt.useable = 1;

    modify_ldt(1, &ldt, sizeof(ldt)); /* write ldt entry */



    modify_ldt(0, &ldt_table, sizeof(ldt_table)); /* read ldt entries */

#if 0

    {

        int i;

        for(i=0;i<3;i++)

            printf("%d: %016Lx\n", i, ldt_table[i]);

    }

#endif

    /* do some tests with fs or gs */

    asm volatile ("movl %0, %%fs" : : "r" (MK_SEL(1)));



    seg_data1[1] = 0xaa;

    seg_data2[1] = 0x55;



    asm volatile ("fs movzbl 0x1, %0" : "=r" (res));

    printf("FS[1] = %02x\n", res);



    asm volatile ("pushl %%gs\n"

                  "movl %1, %%gs\n"

                  "gs movzbl 0x1, %0\n"

                  "popl %%gs\n"

                  : "=r" (res)

                  : "r" (MK_SEL(2)));

    printf("GS[1] = %02x\n", res);



    /* tests with ds/ss (implicit segment case) */

    tmp = 0xa5;

    asm volatile ("pushl %%ebp\n\t"

                  "pushl %%ds\n\t"

                  "movl %2, %%ds\n\t"

                  "movl %3, %%ebp\n\t"

                  "movzbl 0x1, %0\n\t"

                  "movzbl (%%ebp), %1\n\t"

                  "popl %%ds\n\t"

                  "popl %%ebp\n\t"

                  : "=r" (res), "=r" (res2)

                  : "r" (MK_SEL(1)), "r" (&tmp));

    printf("DS[1] = %02x\n", res);

    printf("SS[tmp] = %02x\n", res2);



    segoff.seg = MK_SEL(2);

    segoff.offset = 0xabcdef12;

    asm volatile("lfs %2, %0\n\t"

                 "movl %%fs, %1\n\t"

                 : "=r" (res), "=g" (res2)

                 : "m" (segoff));

    printf("FS:reg = %04x:%08x\n", res2, res);



    TEST_LR("larw", "w", MK_SEL(2), 0x0100);

    TEST_LR("larl", "", MK_SEL(2), 0x0100);

    TEST_LR("lslw", "w", MK_SEL(2), 0);

    TEST_LR("lsll", "", MK_SEL(2), 0);



    TEST_LR("larw", "w", 0xfff8, 0);

    TEST_LR("larl", "", 0xfff8, 0);

    TEST_LR("lslw", "w", 0xfff8, 0);

    TEST_LR("lsll", "", 0xfff8, 0);



    TEST_ARPL("arpl", "w", 0x12345678 | 3, 0x762123c | 1);

    TEST_ARPL("arpl", "w", 0x12345678 | 1, 0x762123c | 3);

    TEST_ARPL("arpl", "w", 0x12345678 | 1, 0x762123c | 1);

}
