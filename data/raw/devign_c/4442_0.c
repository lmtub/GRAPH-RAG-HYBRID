void test_misc(void)

{

    char table[256];

    long res, i;



    for(i=0;i<256;i++) table[i] = 256 - i;

    res = 0x12345678;

    asm ("xlat" : "=a" (res) : "b" (table), "0" (res));

    printf("xlat: EAX=" FMTLX "\n", res);



#if defined(__x86_64__)

#if 0

    {

        /* XXX: see if Intel Core2 and AMD64 behavior really

           differ. Here we implemented the Intel way which is not

           compatible yet with QEMU. */

        static struct __attribute__((packed)) {

            uint64_t offset;

            uint16_t seg;

        } desc;

        long cs_sel;



        asm volatile ("mov %%cs, %0" : "=r" (cs_sel));



        asm volatile ("push %1\n"

                      "call func_lret\n"

                      : "=a" (res)

                      : "r" (cs_sel) : "memory", "cc");

        printf("func_lret=" FMTLX "\n", res);



        desc.offset = (long)&func_lret;

        desc.seg = cs_sel;



        asm volatile ("xor %%rax, %%rax\n"

                      "rex64 lcall *(%%rcx)\n"

                      : "=a" (res)

                      : "c" (&desc)

                      : "memory", "cc");

        printf("func_lret2=" FMTLX "\n", res);



        asm volatile ("push %2\n"

                      "mov $ 1f, %%rax\n"

                      "push %%rax\n"

                      "rex64 ljmp *(%%rcx)\n"

                      "1:\n"

                      : "=a" (res)

                      : "c" (&desc), "b" (cs_sel)

                      : "memory", "cc");

        printf("func_lret3=" FMTLX "\n", res);

    }

#endif

#else

    asm volatile ("push %%cs ; call %1"

                  : "=a" (res)

                  : "m" (func_lret): "memory", "cc");

    printf("func_lret=" FMTLX "\n", res);



    asm volatile ("pushf ; push %%cs ; call %1"

                  : "=a" (res)

                  : "m" (func_iret): "memory", "cc");

    printf("func_iret=" FMTLX "\n", res);

#endif



#if defined(__x86_64__)

    /* specific popl test */

    asm volatile ("push $12345432 ; push $0x9abcdef ; pop (%%rsp) ; pop %0"

                  : "=g" (res));

    printf("popl esp=" FMTLX "\n", res);

#else

    /* specific popl test */

    asm volatile ("pushl $12345432 ; pushl $0x9abcdef ; popl (%%esp) ; popl %0"

                  : "=g" (res));

    printf("popl esp=" FMTLX "\n", res);



    /* specific popw test */

    asm volatile ("pushl $12345432 ; pushl $0x9abcdef ; popw (%%esp) ; addl $2, %%esp ; popl %0"

                  : "=g" (res));

    printf("popw esp=" FMTLX "\n", res);

#endif

}
