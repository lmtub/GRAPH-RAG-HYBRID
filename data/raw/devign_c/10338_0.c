void test_fcmp(double a, double b)

{

    long eflags, fpus;



    fpu_clear_exceptions();

    asm("fcom %2\n"

        "fstsw %%ax\n"

        : "=a" (fpus)

        : "t" (a), "u" (b));

    printf("fcom(%f %f)=%04lx \n",

           a, b, fpus & (0x4500 | FPUS_EMASK));

    fpu_clear_exceptions();

    asm("fucom %2\n"

        "fstsw %%ax\n"

        : "=a" (fpus)

        : "t" (a), "u" (b));

    printf("fucom(%f %f)=%04lx\n",

           a, b, fpus & (0x4500 | FPUS_EMASK));

    if (TEST_FCOMI) {

        /* test f(u)comi instruction */

        fpu_clear_exceptions();

        asm("fcomi %3, %2\n"

            "fstsw %%ax\n"

            "pushf\n"

            "pop %0\n"

            : "=r" (eflags), "=a" (fpus)

            : "t" (a), "u" (b));

        printf("fcomi(%f %f)=%04lx %02lx\n",

               a, b, fpus & FPUS_EMASK, eflags & (CC_Z | CC_P | CC_C));

        fpu_clear_exceptions();

        asm("fucomi %3, %2\n"

            "fstsw %%ax\n"

            "pushf\n"

            "pop %0\n"

            : "=r" (eflags), "=a" (fpus)

            : "t" (a), "u" (b));

        printf("fucomi(%f %f)=%04lx %02lx\n",

               a, b, fpus & FPUS_EMASK, eflags & (CC_Z | CC_P | CC_C));

    }

    fpu_clear_exceptions();

    asm volatile("fxam\n"

                 "fstsw %%ax\n"

                 : "=a" (fpus)

                 : "t" (a));

    printf("fxam(%f)=%04lx\n", a, fpus & 0x4700);

    fpu_clear_exceptions();

}
