void disas(FILE *out, void *code, unsigned long size)

{

    unsigned long pc;

    int count;

    struct disassemble_info disasm_info;

    int (*print_insn)(bfd_vma pc, disassemble_info *info);



    INIT_DISASSEMBLE_INFO(disasm_info, out, fprintf);



    disasm_info.buffer = code;

    disasm_info.buffer_vma = (unsigned long)code;

    disasm_info.buffer_length = size;



#ifdef HOST_WORDS_BIGENDIAN

    disasm_info.endian = BFD_ENDIAN_BIG;

#else

    disasm_info.endian = BFD_ENDIAN_LITTLE;

#endif

#if defined(__i386__)

    disasm_info.mach = bfd_mach_i386_i386;

    print_insn = print_insn_i386;

#elif defined(__x86_64__)

    disasm_info.mach = bfd_mach_x86_64;

    print_insn = print_insn_i386;

#elif defined(_ARCH_PPC)

    print_insn = print_insn_ppc;

#elif defined(__alpha__)

    print_insn = print_insn_alpha;

#elif defined(__sparc__)

    print_insn = print_insn_sparc;

#if defined(__sparc_v8plus__) || defined(__sparc_v8plusa__) || defined(__sparc_v9__)

    disasm_info.mach = bfd_mach_sparc_v9b;

#endif

#elif defined(__arm__)

    print_insn = print_insn_arm;

#elif defined(__MIPSEB__)

    print_insn = print_insn_big_mips;

#elif defined(__MIPSEL__)

    print_insn = print_insn_little_mips;

#elif defined(__m68k__)

    print_insn = print_insn_m68k;

#elif defined(__s390__)

    print_insn = print_insn_s390;

#elif defined(__hppa__)

    print_insn = print_insn_hppa;

#elif defined(__ia64__)

    print_insn = print_insn_ia64;

#else

    fprintf(out, "0x%lx: Asm output not supported on this arch\n",

	    (long) code);

    return;

#endif

    for (pc = (unsigned long)code; size > 0; pc += count, size -= count) {

	fprintf(out, "0x%08lx:  ", pc);

#ifdef __arm__

        /* since data is included in the code, it is better to

           display code data too */

        fprintf(out, "%08x  ", (int)bfd_getl32((const bfd_byte *)pc));

#endif

	count = print_insn(pc, &disasm_info);

	fprintf(out, "\n");

	if (count < 0)

	    break;

    }

}
