int cpu_x86_gen_code(uint8_t *gen_code_buf, int max_code_size, 

                     int *gen_code_size_ptr,

                     uint8_t *pc_start,  uint8_t *cs_base, int flags)

{

    DisasContext dc1, *dc = &dc1;

    uint8_t *pc_ptr;

    uint16_t *gen_opc_end;

    int gen_code_size;

    long ret;

#ifdef DEBUG_DISAS

    struct disassemble_info disasm_info;

#endif

    

    /* generate intermediate code */



    dc->code32 = (flags >> GEN_FLAG_CODE32_SHIFT) & 1;

    dc->ss32 = (flags >> GEN_FLAG_SS32_SHIFT) & 1;

    dc->addseg = (flags >> GEN_FLAG_ADDSEG_SHIFT) & 1;

    dc->f_st = (flags >> GEN_FLAG_ST_SHIFT) & 7;

    dc->cc_op = CC_OP_DYNAMIC;

    dc->cs_base = cs_base;



    gen_opc_ptr = gen_opc_buf;

    gen_opc_end = gen_opc_buf + OPC_MAX_SIZE;

    gen_opparam_ptr = gen_opparam_buf;



    dc->is_jmp = 0;

    pc_ptr = pc_start;

    do {

        ret = disas_insn(dc, pc_ptr);

        if (ret == -1) {

            /* we trigger an illegal instruction operation only if it

               is the first instruction. Otherwise, we simply stop

               generating the code just before it */

            if (pc_ptr == pc_start)

                return -1;

            else

                break;

        }

        pc_ptr = (void *)ret;

    } while (!dc->is_jmp && gen_opc_ptr < gen_opc_end);

    /* we must store the eflags state if it is not already done */

    if (dc->cc_op != CC_OP_DYNAMIC)

        gen_op_set_cc_op(dc->cc_op);

    if (dc->is_jmp != 1) {

        /* we add an additionnal jmp to update the simulated PC */

        gen_op_jmp_im(ret - (unsigned long)dc->cs_base);

    }

    *gen_opc_ptr = INDEX_op_end;



    /* optimize flag computations */

#ifdef DEBUG_DISAS

    if (loglevel) {

        uint8_t *pc;

        int count;



        INIT_DISASSEMBLE_INFO(disasm_info, logfile, fprintf);

#if 0        

        disasm_info.flavour = bfd_get_flavour (abfd);

        disasm_info.arch = bfd_get_arch (abfd);

        disasm_info.mach = bfd_get_mach (abfd);

#endif

        disasm_info.endian = BFD_ENDIAN_LITTLE;

        if (dc->code32)

            disasm_info.mach = bfd_mach_i386_i386;

        else

            disasm_info.mach = bfd_mach_i386_i8086;

        fprintf(logfile, "----------------\n");

        fprintf(logfile, "IN:\n");

        disasm_info.buffer = pc_start;

        disasm_info.buffer_vma = (unsigned long)pc_start;

        disasm_info.buffer_length = pc_ptr - pc_start;

        pc = pc_start;

        while (pc < pc_ptr) {

            fprintf(logfile, "0x%08lx:  ", (long)pc);

            count = print_insn_i386((unsigned long)pc, &disasm_info);

            fprintf(logfile, "\n");

            pc += count;

        }

        fprintf(logfile, "\n");

        

        fprintf(logfile, "OP:\n");

        dump_ops(gen_opc_buf);

        fprintf(logfile, "\n");

    }

#endif



    /* optimize flag computations */

    optimize_flags(gen_opc_buf, gen_opc_ptr - gen_opc_buf);



#ifdef DEBUG_DISAS

    if (loglevel) {

        fprintf(logfile, "AFTER FLAGS OPT:\n");

        dump_ops(gen_opc_buf);

        fprintf(logfile, "\n");

    }

#endif



    /* generate machine code */

    gen_code_size = dyngen_code(gen_code_buf, gen_opc_buf, gen_opparam_buf);

    flush_icache_range((unsigned long)gen_code_buf, (unsigned long)(gen_code_buf + gen_code_size));

    *gen_code_size_ptr = gen_code_size;



#ifdef DEBUG_DISAS

    if (loglevel) {

        uint8_t *pc;

        int count;



        INIT_DISASSEMBLE_INFO(disasm_info, logfile, fprintf);

#if 0        

        disasm_info.flavour = bfd_get_flavour (abfd);

        disasm_info.arch = bfd_get_arch (abfd);

        disasm_info.mach = bfd_get_mach (abfd);

#endif

#ifdef WORDS_BIGENDIAN

        disasm_info.endian = BFD_ENDIAN_BIG;

#else

        disasm_info.endian = BFD_ENDIAN_LITTLE;

#endif        

        disasm_info.mach = bfd_mach_i386_i386;



        pc = gen_code_buf;

        disasm_info.buffer = pc;

        disasm_info.buffer_vma = (unsigned long)pc;

        disasm_info.buffer_length = *gen_code_size_ptr;

        fprintf(logfile, "OUT: [size=%d]\n", *gen_code_size_ptr);

        while (pc < gen_code_buf + *gen_code_size_ptr) {

            fprintf(logfile, "0x%08lx:  ", (long)pc);

            count = print_insn_i386((unsigned long)pc, &disasm_info);

            fprintf(logfile, "\n");

            pc += count;

        }

        fprintf(logfile, "\n");

        fflush(logfile);

    }

#endif

    return 0;

}
