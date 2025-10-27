static void write_bootloader(uint8_t *base, int64_t run_addr,

                             int64_t kernel_entry)

{

    uint32_t *p;



    /* Small bootloader */

    p = (uint32_t *)base;



    stl_p(p++, 0x08000000 |                                      /* j 0x1fc00580 */

                 ((run_addr + 0x580) & 0x0fffffff) >> 2);

    stl_p(p++, 0x00000000);                                      /* nop */



    /* YAMON service vector */

    stl_p(base + 0x500, run_addr + 0x0580);      /* start: */

    stl_p(base + 0x504, run_addr + 0x083c);      /* print_count: */

    stl_p(base + 0x520, run_addr + 0x0580);      /* start: */

    stl_p(base + 0x52c, run_addr + 0x0800);      /* flush_cache: */

    stl_p(base + 0x534, run_addr + 0x0808);      /* print: */

    stl_p(base + 0x538, run_addr + 0x0800);      /* reg_cpu_isr: */

    stl_p(base + 0x53c, run_addr + 0x0800);      /* unred_cpu_isr: */

    stl_p(base + 0x540, run_addr + 0x0800);      /* reg_ic_isr: */

    stl_p(base + 0x544, run_addr + 0x0800);      /* unred_ic_isr: */

    stl_p(base + 0x548, run_addr + 0x0800);      /* reg_esr: */

    stl_p(base + 0x54c, run_addr + 0x0800);      /* unreg_esr: */

    stl_p(base + 0x550, run_addr + 0x0800);      /* getchar: */

    stl_p(base + 0x554, run_addr + 0x0800);      /* syscon_read: */





    /* Second part of the bootloader */

    p = (uint32_t *) (base + 0x580);



    if (semihosting_get_argc()) {

        /* Preserve a0 content as arguments have been passed */

        stl_p(p++, 0x00000000);                         /* nop */

    } else {

        stl_p(p++, 0x24040002);                         /* addiu a0, zero, 2 */

    }

    stl_p(p++, 0x3c1d0000 | (((ENVP_ADDR - 64) >> 16) & 0xffff)); /* lui sp, high(ENVP_ADDR) */

    stl_p(p++, 0x37bd0000 | ((ENVP_ADDR - 64) & 0xffff));        /* ori sp, sp, low(ENVP_ADDR) */

    stl_p(p++, 0x3c050000 | ((ENVP_ADDR >> 16) & 0xffff));       /* lui a1, high(ENVP_ADDR) */

    stl_p(p++, 0x34a50000 | (ENVP_ADDR & 0xffff));               /* ori a1, a1, low(ENVP_ADDR) */

    stl_p(p++, 0x3c060000 | (((ENVP_ADDR + 8) >> 16) & 0xffff)); /* lui a2, high(ENVP_ADDR + 8) */

    stl_p(p++, 0x34c60000 | ((ENVP_ADDR + 8) & 0xffff));         /* ori a2, a2, low(ENVP_ADDR + 8) */

    stl_p(p++, 0x3c070000 | (loaderparams.ram_low_size >> 16));     /* lui a3, high(ram_low_size) */

    stl_p(p++, 0x34e70000 | (loaderparams.ram_low_size & 0xffff));  /* ori a3, a3, low(ram_low_size) */



    /* Load BAR registers as done by YAMON */

    stl_p(p++, 0x3c09b400);                                      /* lui t1, 0xb400 */



#ifdef TARGET_WORDS_BIGENDIAN

    stl_p(p++, 0x3c08df00);                                      /* lui t0, 0xdf00 */

#else

    stl_p(p++, 0x340800df);                                      /* ori t0, r0, 0x00df */

#endif

    stl_p(p++, 0xad280068);                                      /* sw t0, 0x0068(t1) */



    stl_p(p++, 0x3c09bbe0);                                      /* lui t1, 0xbbe0 */



#ifdef TARGET_WORDS_BIGENDIAN

    stl_p(p++, 0x3c08c000);                                      /* lui t0, 0xc000 */

#else

    stl_p(p++, 0x340800c0);                                      /* ori t0, r0, 0x00c0 */

#endif

    stl_p(p++, 0xad280048);                                      /* sw t0, 0x0048(t1) */

#ifdef TARGET_WORDS_BIGENDIAN

    stl_p(p++, 0x3c084000);                                      /* lui t0, 0x4000 */

#else

    stl_p(p++, 0x34080040);                                      /* ori t0, r0, 0x0040 */

#endif

    stl_p(p++, 0xad280050);                                      /* sw t0, 0x0050(t1) */



#ifdef TARGET_WORDS_BIGENDIAN

    stl_p(p++, 0x3c088000);                                      /* lui t0, 0x8000 */

#else

    stl_p(p++, 0x34080080);                                      /* ori t0, r0, 0x0080 */

#endif

    stl_p(p++, 0xad280058);                                      /* sw t0, 0x0058(t1) */

#ifdef TARGET_WORDS_BIGENDIAN

    stl_p(p++, 0x3c083f00);                                      /* lui t0, 0x3f00 */

#else

    stl_p(p++, 0x3408003f);                                      /* ori t0, r0, 0x003f */

#endif

    stl_p(p++, 0xad280060);                                      /* sw t0, 0x0060(t1) */



#ifdef TARGET_WORDS_BIGENDIAN

    stl_p(p++, 0x3c08c100);                                      /* lui t0, 0xc100 */

#else

    stl_p(p++, 0x340800c1);                                      /* ori t0, r0, 0x00c1 */

#endif

    stl_p(p++, 0xad280080);                                      /* sw t0, 0x0080(t1) */

#ifdef TARGET_WORDS_BIGENDIAN

    stl_p(p++, 0x3c085e00);                                      /* lui t0, 0x5e00 */

#else

    stl_p(p++, 0x3408005e);                                      /* ori t0, r0, 0x005e */

#endif

    stl_p(p++, 0xad280088);                                      /* sw t0, 0x0088(t1) */



    /* Jump to kernel code */

    stl_p(p++, 0x3c1f0000 | ((kernel_entry >> 16) & 0xffff));    /* lui ra, high(kernel_entry) */

    stl_p(p++, 0x37ff0000 | (kernel_entry & 0xffff));            /* ori ra, ra, low(kernel_entry) */

    stl_p(p++, 0x03e00009);                                      /* jalr ra */

    stl_p(p++, 0x00000000);                                      /* nop */



    /* YAMON subroutines */

    p = (uint32_t *) (base + 0x800);

    stl_p(p++, 0x03e00009);                                     /* jalr ra */

    stl_p(p++, 0x24020000);                                     /* li v0,0 */

    /* 808 YAMON print */

    stl_p(p++, 0x03e06821);                                     /* move t5,ra */

    stl_p(p++, 0x00805821);                                     /* move t3,a0 */

    stl_p(p++, 0x00a05021);                                     /* move t2,a1 */

    stl_p(p++, 0x91440000);                                     /* lbu a0,0(t2) */

    stl_p(p++, 0x254a0001);                                     /* addiu t2,t2,1 */

    stl_p(p++, 0x10800005);                                     /* beqz a0,834 */

    stl_p(p++, 0x00000000);                                     /* nop */

    stl_p(p++, 0x0ff0021c);                                     /* jal 870 */

    stl_p(p++, 0x00000000);                                     /* nop */

    stl_p(p++, 0x08000205);                                     /* j 814 */

    stl_p(p++, 0x00000000);                                     /* nop */

    stl_p(p++, 0x01a00009);                                     /* jalr t5 */

    stl_p(p++, 0x01602021);                                     /* move a0,t3 */

    /* 0x83c YAMON print_count */

    stl_p(p++, 0x03e06821);                                     /* move t5,ra */

    stl_p(p++, 0x00805821);                                     /* move t3,a0 */

    stl_p(p++, 0x00a05021);                                     /* move t2,a1 */

    stl_p(p++, 0x00c06021);                                     /* move t4,a2 */

    stl_p(p++, 0x91440000);                                     /* lbu a0,0(t2) */

    stl_p(p++, 0x0ff0021c);                                     /* jal 870 */

    stl_p(p++, 0x00000000);                                     /* nop */

    stl_p(p++, 0x254a0001);                                     /* addiu t2,t2,1 */

    stl_p(p++, 0x258cffff);                                     /* addiu t4,t4,-1 */

    stl_p(p++, 0x1580fffa);                                     /* bnez t4,84c */

    stl_p(p++, 0x00000000);                                     /* nop */

    stl_p(p++, 0x01a00009);                                     /* jalr t5 */

    stl_p(p++, 0x01602021);                                     /* move a0,t3 */

    /* 0x870 */

    stl_p(p++, 0x3c08b800);                                     /* lui t0,0xb400 */

    stl_p(p++, 0x350803f8);                                     /* ori t0,t0,0x3f8 */

    stl_p(p++, 0x91090005);                                     /* lbu t1,5(t0) */

    stl_p(p++, 0x00000000);                                     /* nop */

    stl_p(p++, 0x31290040);                                     /* andi t1,t1,0x40 */

    stl_p(p++, 0x1120fffc);                                     /* beqz t1,878 <outch+0x8> */

    stl_p(p++, 0x00000000);                                     /* nop */

    stl_p(p++, 0x03e00009);                                     /* jalr ra */

    stl_p(p++, 0xa1040000);                                     /* sb a0,0(t0) */



}
